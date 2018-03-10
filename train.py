# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

import os
import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data
import model

use_gpu = torch.cuda.is_available()

def main():
    print("Preparing to synthesize training data...")

    # Backgrounds and fonts for synthesizing the dataset
    bg_dir = '/media/Data/SUN397/'
    font_dir = '/media/Data/fonts/'

    bg_fnames = []
    for root, dirs, filenames in os.walk(bg_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                bg_fnames.append(os.path.join(root, filename))
    
    font_fnames = []
    for root, dirs, filenames in os.walk(font_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() == '.ttf':
                font_fnames.append(os.path.join(root, filename))

    dataset = data.PlateDataset(font_fnames, bg_fnames)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, drop_last=True, pin_memory=True)

    # Neural net models
    plate_encoder = model.PlateEncoder()
    plate_decoder = model.PlateDecoder()

    # Save a few sample training images
    os.system("rm -f tmp/sample*.png")
    for i in range(20):
        x, y, _, _ = dataset[i]
        im = Image.fromarray(x.numpy())
        im.save('tmp/sample%d_%d.png' % (i, y))

    # Load an image to test periodically
    test_big_x = torch.from_numpy(np.array(Image.open('samples/test_0.png').convert('L')))
    test_big_x = (test_big_x.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor) - 127) / 128.0
    test_big_x = Variable(test_big_x, requires_grad=False)

    test_plate_x = torch.from_numpy(np.array(Image.open('samples/plate_0.png').convert('L')))
    test_plate_x = (test_plate_x.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor) - 127) / 128.0
    test_plate_x = Variable(test_plate_x, requires_grad=False)

    test_y_stim = Variable(torch.LongTensor([[data.start_token]*data.num_steps]), requires_grad=False)

    if use_gpu:
        plate_encoder.cuda()
        plate_decoder.cuda()
        test_big_x = test_big_x.cuda()
        test_plate_x = test_plate_x.cuda()
        test_y_stim = test_y_stim.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': plate_encoder.parameters()},
        {'params': plate_decoder.parameters()}
    ])

    print('Training...')
    for it, (x, y_presence, y_stim, y_exp) in enumerate(data_loader):
        x = (x.unsqueeze(1).type(torch.FloatTensor) - 127) / 128.0
        batch_size, num_steps = y_exp.size()

        if use_gpu:
            x = x.cuda()
            y_presence = y_presence.cuda()
            y_stim = y_stim.cuda()
            y_exp = y_exp.cuda()
        
        x = Variable(x, requires_grad=False)
        y_presence = Variable(y_presence, requires_grad=False)
        y_stim = Variable(y_stim, requires_grad=False)
        y_exp = Variable(y_exp, requires_grad=False)

        plate_encoder.train()
        plate_decoder.train()

        optimizer.zero_grad()

        f, y_presence_pred_scores = plate_encoder(x)
        loss = criterion(y_presence_pred_scores.squeeze(3).squeeze(2), y_presence)
        y_pred_scores, _ = plate_decoder(y_stim, f, h=None, teacher_forcing=True)
        for i in range(num_steps):
            loss += criterion(y_pred_scores[:, i, :], y_exp[:, i])
        loss.backward()

        optimizer.step()

        if it % 100 == 0:
            print 'Iteration %d -- loss: %.4f' % (it, loss)
            if it % 1000 == 0:
                torch.save(plate_encoder.state_dict(), "params/encoder.dat")
                torch.save(plate_decoder.state_dict(), "params/decoder.dat")

                plate_encoder.eval()
                plate_decoder.eval()

                # test the plate presence detector on larger image
                _, test_big_y_presence_pred_scores = plate_encoder(test_big_x)
                test_big_y_pred = F.softmax(test_big_y_presence_pred_scores, dim=1)
                test_big_y_img = (test_big_y_pred[:, 0, :, :] > 0.95)*255

                im = Image.fromarray(test_big_y_img.squeeze().cpu().data.numpy())
                print("Saving sample output...\n")
                im.save('tmp/op_big.png')

                # test the plate recognizer on small image having one number plate
                test_plate_f, _ = plate_encoder(test_plate_x)
                test_plate_y_pred_scores, _ = plate_decoder(test_y_stim, test_plate_f, h=None, teacher_forcing=False)
                _, num_steps = test_y_stim.size()
                for step in range(num_steps):
                    val, test_plate_o = torch.max(F.softmax(test_plate_y_pred_scores[:, step, :], dim=1), dim=1)
                    idx = int(test_plate_o)
                    print "(%s, %.2f)\n" % (data.tokens[idx], val)

if __name__ == '__main__':
    main()
