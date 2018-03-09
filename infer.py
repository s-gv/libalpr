# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data
import model
import cc

use_gpu = torch.cuda.is_available()

plate_encoder = model.PlateEncoder()
plate_decoder = model.PlateDecoder()

plate_encoder.load_state_dict(torch.load("params/encoder.dat"))
plate_decoder.load_state_dict(torch.load("params/decoder.dat"))

plate_encoder.eval()
plate_decoder.eval()

def find_plates(img, min_p=0.3, dbg=False):
    ''' Find number plates in the image
        img is a PIL image
        return list of ((x0, y0), (x1, y1), "PLATE_CHARS")
    '''

    test_x = torch.from_numpy(np.array(img.convert('L')))
    test_x = (test_x.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor) - 127) / 128.0
    test_x = Variable(test_x, requires_grad=False)
    
    test_y_stim = Variable(torch.LongTensor([[data.start_token]*data.num_steps]), requires_grad=False)

    if use_gpu:
        plate_encoder.cuda()
        plate_decoder.cuda()
        test_x = test_x.cuda()
        test_y_stim = test_y_stim.cuda()

    f, p = plate_encoder(test_x)
    test_y_presence_img = (F.softmax(p, dim=1)[:, 0, :, :] > 0.95)*255
    _, f_c, f_h, f_w = f.size()

    if dbg:
        im = Image.fromarray(test_y_presence_img.squeeze().cpu().data.numpy())
        print "Saving intermediate localization result..."
        im.save('tmp/op_loc.png')

    test_y_idxs = torch.nonzero(test_y_presence_img.squeeze())
    if len(test_y_idxs.size()) == 0:
        test_y_idxs = []
        
    plates = []
    f_crops = []
    hot_spots = [(int(p[1]), int(p[0])) for p in test_y_idxs]
    for i, (xc, yc) in enumerate(cc.find_ccs(hot_spots)):
        if i > 10:
            # max 10 plates per image
            break
        if xc < (f_w - plate_decoder.f_w) and yc < (f_h - plate_decoder.f_h/2):
            if dbg:
                print "\n", (xc, yc)
                plate_img = img.crop((xc*8, yc*8, xc*8 + 94, yc*8 + 54))
                plate_img.save("tmp/op_plate_%d.png" % i)
            
            xstart, xstop = xc, xc + plate_decoder.f_w
            ystart, ystop = yc, yc + plate_decoder.f_h
            f_crop = f[:, :, ystart:ystop, xstart:xstop]

            plate = ((xc*8, yc*8), (xc*8 + 94, yc*8 + 54))
            f_crops.append(f_crop)
            plates.append(plate)

    test_y_stim = test_y_stim.expand(len(f_crops), -1).contiguous()
    f_crops = torch.cat(f_crops, dim=0)
    test_y_pred_scores, _ = plate_decoder(test_y_stim, f_crops, h=None, teacher_forcing=False)
    test_y_pred = F.softmax(test_y_pred_scores, dim=2)
    _, num_steps = test_y_stim.size()

    filtered_plates = []
    for i in range(len(plates)):
        chs = []
        for step in range(num_steps):
            val, test_plate_o = torch.max(test_y_pred[i, step, :], dim=0)
            idx, val = int(test_plate_o), float(val)
            if dbg:
                print "(%s, %.2f)\n" % (data.tokens[idx], val)
            chs.append((data.tokens[idx] if idx < data.start_token else '.', val))
        if chs[0][0] != '.' and min(val for _, val in chs) > min_p:
            filtered_plates.append((plates[i][0], plates[i][1], ''.join(ch for ch, _ in chs)))
    
    return filtered_plates
