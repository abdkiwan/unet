
'''
This script allows the user to segment a new image using a trained model.
The user has to provide the network parameters as a list of arguments.
'''

from __future__ import print_function

from matplotlib import cm
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import numpy as np
import argparse
import glob
import os

from utils import load_img, save_img
from model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_dir',
                    help = 'Directory of the images to be predicted.')

parser.add_argument('--model_save_dir',
                    help = 'Directory of the trained model used for prediction.')

parser.add_argument('--img_size', default = 240,
                    help = 'Dimension of the image used for prediction (one value).', type = int)
       
def main():
    
    FLAGS = parser.parse_args()

    # Calculate the predictions for all the images in the input_img_dir.
    for img_name in os.listdir(FLAGS.input_img_dir):

        if img_name.endswith('.png'):
            original_img = load_img(FLAGS.input_img_dir+'/'+img_name)   # load_img function exists in file utils.py

            # Resizing image because of the small memory size
            input_img = tl.prepro.imresize(original_img, [FLAGS.img_size, FLAGS.img_size])
            input_img = np.reshape(input_img, [1, FLAGS.img_size, FLAGS.img_size, 3])

            unet = UNet(FLAGS.img_size)

            # The output is an array of the size(img_size * img_size, 1)
            prediction = unet.predict(input_img, FLAGS.model_save_dir)

            # Saving the image given the probabilities
            # save_img fnuction exists in file utils.py
            save_img(prediction, original_img, FLAGS.img_size, FLAGS.input_img_dir+'/'+img_name.split('.')[0]+'_pred.png')


if __name__ == '__main__':
    main()
