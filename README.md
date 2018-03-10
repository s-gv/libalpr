libalpr
=======

libalpr locates and recognizes license plates in images.
A convnet is used to generate a feature map of the image, and an LSTM
reads the license plate character-by-character by paying attention to
the part of the feature map corresponding to each character. The network
architecture is fully convolutional, so it can accept images of arbitrary 
size, but it looks for number plates in windows of size 94x54 px.

How-to
------

Try libalpr with the pre-trained model: `python demo.py`

To train the model, the number plates are synthetically generated,
but font files and background images are required. Specify their
location in `train.py`.

Requirements
------------

- Python 2.7
- PyTorch 0.3
- Pillow
- Numpy

