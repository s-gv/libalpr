# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

import Image
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data


class PlateEncoder(nn.Module):
    def __init__(self):
        super(PlateEncoder, self).__init__()

        self.f_c = 512 # num_channels in output feature map
        self.f_w, self.f_h = (8, 3) # Input window of size 94x54 reduced to 8x3 after convs
        
        self.conv_layers = torch.nn.Sequential( # Ip: 94x54
            nn.Conv2d(1, 32, (3, 3)), nn.ReLU(), # Op: 92x52
            nn.MaxPool2d((2, 2)), # Op: 46x26
            nn.Conv2d(32, 64, (3, 3)), nn.ReLU(), # Op: 44x24
            nn.MaxPool2d((2, 2)), # Op: 22x12
            nn.Conv2d(64, 128, (3, 3)), nn.ReLU(), # Op: 20x10
            nn.MaxPool2d((2, 2)), # Op: 10x5
            nn.Conv2d(128, 256, (3, 3)), nn.ReLU(), # Op: 8x3
            nn.BatchNorm2d(256),
            nn.Conv2d(256, self.f_c, (1, 1)), # Op: 8x3
        )

        self.presence_layers = torch.nn.Sequential( # Ip: 8x3
            nn.Conv2d(self.f_c, 64, (self.f_h, self.f_w)), nn.ReLU(), # Op: 1x1
            nn.Conv2d(64, 2, (1, 1)), # Op: 1x1
        )
        

    def forward(self, x):
        f = self.conv_layers(x)
        p = self.presence_layers(f)
        return f, p


class PlateDecoder(nn.Module):
    def __init__(self):
        super(PlateDecoder, self).__init__()

        self.is_dbg_attn = False

        self.num_tokens = data.num_tokens
        self.embedding_size = 32
        self.hidden_size = 64
        self.num_layers = 1
        self.f_c= 512
        self.wf_c = 128
        self.f_w, self.f_h = 8, 3

        self.embedding = nn.Embedding(self.num_tokens, self.embedding_size)
        self.decoder_rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)

        self.h_attn_layer = nn.Linear(self.hidden_size, self.wf_c)
        self.f_attn_layer = nn.Conv2d(self.f_c + self.f_w + self.f_h, self.wf_c, (1, 1))
        self.attn_layer = nn.Conv2d(self.wf_c, 1, (1, 1))

        self.rnn_output_layer = nn.Linear(self.hidden_size, self.num_tokens)
        self.img_output_layer = nn.Linear(self.f_c, self.num_tokens)

    def init_hidden(self, batch_size, use_gpu=True):
        init_h = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))
        init_c = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_size)))
        if use_gpu:
            init_h = init_h.cuda()
            init_c = init_c.cuda()
        return init_h, init_c

    def pos_pad(self, batch_size, use_gpu=True):
        x_pad = torch.eye(self.f_w).view(1, self.f_w, 1, self.f_w).expand(batch_size, -1, self.f_h, -1)
        y_pad = torch.eye(self.f_h).view(1, self.f_h, self.f_h, 1).expand(batch_size, -1, -1, self.f_w)

        pad = Variable(torch.cat([x_pad, y_pad], dim=1), requires_grad=False)
        if use_gpu:
            pad = pad.cuda()
        
        return pad

    def forward(self, x, f, h=None, teacher_forcing=False):
        ''' f (feature map) is a FloatTensor of size (batch_size, f_c, f_h, f_w)
            x is a LongTensor of size (batch_size, num_steps)
            h is the hidden state got from this function or init_hidden()

            returns (o, h_out) where
            o is a FloatTensor or size (batch_size, num_steps, num_tokens)
            h_out is the hidden state after all the steps are done
        '''
        batch_size, f_c, f_h, f_w = f.size()
        _, num_steps = x.size()
        assert (f_c == self.f_c) and (f_h == self.f_h) and (f_w == self.f_w)

        f_padded = torch.cat([f, self.pos_pad(batch_size, use_gpu=x.is_cuda)], dim=1)
        wf = self.f_attn_layer(f_padded)

        if h == None:
            h = self.init_hidden(batch_size, use_gpu=f.is_cuda)

        ops = []
        for i in range(num_steps):
            xi = x[:, i]
            if i > 0 and not teacher_forcing:
                _, xi = torch.max(ops[-1], dim=1)
            
            xe = self.embedding(xi).unsqueeze(1) # (batch_size, 1, embedding_size)
            o, h = self.decoder_rnn(xe, h)

            hidden = h[0]
            wh = self.h_attn_layer(hidden).view(batch_size, self.wf_c, 1, 1).expand(-1, -1, f_h, f_w)

            a = self.attn_layer(torch.tanh(wh + wf)) # (batch_size, 1, f_h, f_w)
            alpha = F.softmax(a.view(batch_size, -1), dim=1).view(batch_size, 1, f_h, f_w)

            if self.is_dbg_attn and batch_size == 1:
                # visualize attention
                im = Image.fromarray((alpha*255).byte().squeeze(0).squeeze(0).cpu().data.numpy())
                im.save('tmp/attn_%02d.png' % i)
                #for j in range(f_h):
                    #print ', '.join("%.2f" % alpha[0, 0, j, i] for i in range(f_w))

            u = (f * alpha).view(batch_size, f_c, -1).sum(2)

            wo = self.rnn_output_layer(o.squeeze(1))
            wu = self.img_output_layer(u)

            op = (wo + wu)
            ops.append(op)

        return torch.cat([op.view(batch_size, 1, self.num_tokens) for op in ops], dim=1), h

