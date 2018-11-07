
'''
The script contains some general functions that are used by the other files.
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorlayer as tl
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import csv
import os

def augment_img(img, img_size):

	"""
    Applying some transformations on image.
    ----------
    Args:
        img: the image to be transformed.
        img_size: size of image to be resized to.
    Returns:
        a list of the transformations applied to the given image.
    """

	augmented_img = []
	
	augmented_img.append(tl.prepro.imresize(tl.prepro.flip_axis(img, axis=0), [img_size, img_size]))
	augmented_img.append(tl.prepro.imresize(tl.prepro.flip_axis(img, axis=1), [img_size, img_size]))
	augmented_img.append(tl.prepro.imresize(tl.prepro.rotation(img, rg=90, fill_mode='nearest'), [img_size, img_size]))
	augmented_img.append(tl.prepro.imresize(tl.prepro.shift(img, wrg=0.10, hrg=0.10, fill_mode='nearest'), [img_size, img_size]))
	augmented_img.append(tl.prepro.imresize(tl.prepro.elastic_transform(img, alpha=720, sigma=24), [img_size, img_size]))

	return augmented_img


def load_data(data_dir, img_size, augment_data = False):

	"""
    Loading dataset.
    ----------
    Args:
        data_dir : the directory of the images.
        img_size : to resize the images.
        augment_data : to determine whether to augment the images.
    Returns:
    	Two lists of images and the corresponding labels
    """

    # The lists to be returned after being filled.
	X = []
	y = []
	
	for data_phase in os.listdir(data_dir):
		
		processed_imgs = []

		# to search for all the images in the directory
		for img_name in os.listdir(data_dir+'/'+data_phase):
			img_name = img_name.split('.')[0]

			# to check if the image has been processed before.
			if img_name not in processed_imgs:
				processed_imgs.append(img_name)

				img_dir = data_dir+'/'+data_phase+'/'+img_name+'.png'
				label_dir = data_dir+'/'+data_phase+'/'+img_name+'.csv'
				
				# loading the image and the label.
				img = load_img(img_dir)
				original_img_size = img.shape[0]
				label = load_label(label_dir, original_img_size)

				X.append(tl.prepro.imresize(img, [img_size, img_size]))
				y.append(tl.prepro.imresize(label, [img_size, img_size]))
				if augment_data:
					augmented_img = augment_img(img, img_size)
					augmented_label = augment_img(label, img_size)

					X = X + augmented_img
					y = y + augmented_label	
	
	return np.asarray(X), np.asarray(y)


def load_img(img_dir):
	"""
    Reading an image.
    ----------
    Args:
        img_dir: dir of the image.
    Returns:
        a numpy array of the image.
    """
	img = Image.open(img_dir)
	img = np.array(img, np.float32)

	return img


def save_img(prediction, input_img, reduced_img_size, output_img_dir):

	"""
    Saving the image with the predicted mitosis colored in yellow.
    ----------
    Args:
        prediction : the probabilities resulted by the neural network.
        input_img.
        reduced_img_size : the image size used in training.
        output_img_dir : the directory of saving the output image.
    """

	original_img_size = input_img.shape[0]

    # reshaping the array of the prediction.
	predicted_img = np.reshape(prediction, [reduced_img_size, reduced_img_size])
	
	
	predicted_img = Image.fromarray(predicted_img.astype('uint8'))  # Converting numpy array into image.

    # resize the image into the size of the input image
	predicted_img = predicted_img.resize((original_img_size, original_img_size), Image.NEAREST)

	# Turn the pixel color in the input image into yellow in the locations of the detected mitosis.
	for r in range(0, original_img_size):
		for c in range(0, original_img_size):		
			if predicted_img.getpixel((r, c)) == 1: input_img[r, c] = [255, 255, 0]
			

	input_img = Image.fromarray(input_img.astype('uint8'))   # Converting numpy array into image.
	input_img.save(output_img_dir)


def load_label(label_dir, original_size):
	
	"""
    Creating a ground truth (label) image given the locations of the mitosis as a csv file.
    ----------
    Args:
        label_dir : dir of csv file.
        original_size : size of the image to be created.
    Returns:
    	a numpy array of the ground truth (label)
    """

    # initializing the label image.
	label_arr = [[0 for i in range(original_size)] for j in range(original_size)]
	with open(label_dir) as csvfile:
	    readCSV = csv.reader(csvfile, delimiter=',')
	    for row in readCSV:
	        
	        for i in range(0, len(row)-1, 2):
	        	y = int(row[i])
	        	x = int(row[i+1])
	        	label_arr[x][y] = 1
	
	# Transforming the array into numpy array, and reshaping it.
	label_arr = np.array(label_arr, np.float32)
	label_arr = np.reshape(label_arr, (original_size, original_size, 1))

	return label_arr

def plot_loss(train_loss_values, test_loss_values, save_dir):

	"""
    Plotting the train and test loss on the same figure.
    ----------
    Args:
        train_loss_values.
        test_loss_values.
        save_dir : dir of saving the figure.
    """

    # creating a list of the epochs according to the length of the given train_loss_values
	epochs = range(1, len(train_loss_values)+1)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(epochs, train_loss_values)
	ax.plot(epochs, test_loss_values)
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Loss')
	plt.savefig(save_dir)
	plt.show()