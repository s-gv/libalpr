# Copyright (c) 2018 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by the AGPLv3 license that can be
# found in the LICENSE file.

import os
from PIL import Image, ImageDraw
import infer

def main():
    os.system("rm -f tmp/sample*.png tmp/op*.png")
    img = Image.open("samples/test.png").convert('RGBA')
    plates = infer.find_plates(img, dbg=True)
    print plates

    draw = ImageDraw.Draw(img)
    for plate in plates:
        p_lt, p_rb, chars = plate
        draw.rectangle((p_lt, p_rb), outline=(255, 0, 0))
        draw.text((p_lt[0], p_lt[1]-15), chars, fill=(255, 0, 0))
    img.save("tmp/op_plates.png")

if __name__ == '__main__':
    main()