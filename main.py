
'''
This script allows the user to train U-Net.
The user has to provide the network parameters as a list of arguments.
'''

import tensorflow as tf
import numpy as np
import argparse
import io
import PIL

from utils import load_data, plot_loss
from model import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default = 'data',
					help = 'Directory for dataset.')

parser.add_argument('--img_size', type = int, default = 240,
					help = 'Dimension of the image used for training (one value).')

parser.add_argument('--learning_rate',  default = 0.00001,
					help = 'Learning rate for optimizer', type = float)

parser.add_argument('--num_epochs', default = 10,
					help = 'Number of epochs for training', type = int) 

parser.add_argument('--augment_data', default = False,
					help = 'Whether to apply data augmentation to the original dataset.', type = bool)

parser.add_argument('--model_save_dir', default='checkpoints/model',
					help = 'Directory of saving the trained model.')

def main():

	FLAGS = parser.parse_args()

	# Loading train and test data.
	# load_data function exists in file utils.py
	print("Loading dataset.")
	X_train, y_train = load_data(FLAGS.dataset_dir+'/train', FLAGS.img_size, FLAGS.augment_data)
	X_test, y_test = load_data(FLAGS.dataset_dir+'/test', FLAGS.img_size, FLAGS.augment_data)
	
	# Making sure that the data was loaded successfully.
	print("Train set image size : ", X_train.shape)
	print("Train set label size : ", y_train.shape)
	print("Test set image size : ", X_test.shape)
	print("Test set label size : ", y_test.shape)
	print("Dataset loaded successfully.")
	
	# Creating a unet object.
	# class UNet exists in file model.py
	unet = UNet(FLAGS.img_size)
	
	# Training the network, printing the loss value for every epoch
	# , and the accuracy on the test set after the training is complete
	train_loss_values, test_loss_values = unet.train(X_train, y_train, X_test, y_test, 
		FLAGS.num_epochs, FLAGS.learning_rate, FLAGS.model_save_dir)

	# Plotting loss values on train and test set, and saving it as an image Loss.png
	# plot_loss exists in file utils.py
	plot_loss(train_loss_values, test_loss_values, 'Loss.png')

if __name__ == '__main__':
	main()
