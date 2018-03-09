# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

import os
import random
import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import torch
import torch.utils.data

start_token = 36
end_token = 37
tokens = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["<START>", "<END>"]

class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, font_fnames, bg_fnames):
        super(PlateDataset, self).__init__()

        self.fonts = [ImageFont.truetype(f_font, size) for f_font in font_fnames for size in range(20, 36)]
        self.bg_imgs = [Image.open(random.choice(bg_fnames)).convert('L') for _ in range(10)]
    
    def __len__(self):
        return 11423000

    def draw_text(self, img, txt, font, txt_color, x_spacing=0, offset_x=0, offset_y=0):
        draw = ImageDraw.Draw(img)
        mask = font.getmask(txt)
        width, height = mask.size[0] + (len(txt)-1) * x_spacing, mask.size[1]
        xstart, ystart = (img.size[0] - width)/2 + offset_x, (img.size[1] - height)/2 + offset_y
        for idx, c in enumerate(txt):
            mask = font.getmask(c)
            if mask.size[1] > 0:
                mask_img = Image.new('L', mask.size)
                mask_img.putdata(mask)

                txt_color_img = Image.new('L', mask.size, 255)
                mask_img = Image.composite(txt_color_img, Image.new('L', mask.size, 0), mask_img)
                draw.bitmap((xstart, ystart), mask_img, txt_color)
            
            xstart += mask.size[0] + x_spacing

    def __getitem__(self, idx):
        is_valid = random.choice([True, False])
        y_is_present = 1 if is_valid else 0

        n_ch = 10
        labels = [end_token for _ in range(n_ch)]
        

        bg_color = random.randint(100, 255)
        txt_color = random.randint(0, 70)

        img = Image.new('L', (376, 216), color=bg_color)
        img_w, img_h = img.size
        targ_w, targ_h = 188, 108

        if not is_valid or random.choice([True, True, False]):
            bg_img = random.choice(self.bg_imgs)
            bg_w, bg_h = bg_img.size
            if bg_w > img_w and bg_h > img_h:
                off_x = random.randint(0, bg_w-img_w-1)
                off_y = random.randint(0, bg_h-img_h-1)
                img = bg_img.crop((off_x, off_y, off_x+img_w, off_y+img_h))

        if is_valid:
            labels = [random.randint(0, start_token-1) for _ in range(n_ch)]

            state = random.choice(["KA", "TN", "GJ", "MH"])
            labels[0:2] = tokens.index(state[0]), tokens.index(state[1])
            labels[2:4] = random.randint(0, 9), random.randint(0, 9)
            labels[4:6] = random.randint(10, 35), random.randint(10, 35)
            labels[6:10] = [random.randint(0, 9) for _ in range(4)]

            line1_sep = ""
            if random.choice([True, False]):
                line1_sep = "-"
            elif random.choice([True, False]):
                line1_sep = " "

            line2_sep = " " if random.choice([True, False]) else ""

            line1 = ''.join(tokens[labels[i]] + (line1_sep if i == 1 else "") for i in range(4))
            line2 = ''.join(tokens[labels[i]] + (line2_sep if i == 5 else "") for i in range(4, n_ch))
            
            font = random.choice(self.fonts)
            x_spacing = random.randint(0, 4)
            y_spacing = random.randint(10, 15)
            self.draw_text(img, line1, font, txt_color, x_spacing=x_spacing, offset_y=-y_spacing)
            self.draw_text(img, line2, font, txt_color, x_spacing=x_spacing, offset_y=y_spacing)

        rot = random.uniform(-30.0, 30.0)
        img = img.rotate(rot, resample=Image.BILINEAR)

        start_x = (img_w - targ_w)/2
        start_y = (img_h - targ_h)/2
        offset_x = random.choice([1, -1]) * random.randint(0, 16)
        offset_y = random.choice([1, -1]) * random.randint(0, 16)
        img = img.crop((start_x+offset_x, start_y+offset_y, start_x+offset_x+targ_w, start_y+offset_y+targ_h))
        img.thumbnail((targ_w/2, targ_h/2), Image.BILINEAR)

        np_img = np.array(img)

        y_stim = torch.LongTensor([start_token] + labels)
        y_exp = torch.LongTensor(labels + [end_token])

        return torch.from_numpy(np_img), y_is_present, y_stim, y_exp