#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:
import numpy as np
import cv2

# Add any python libraries here

def loadImages(folder_name, image_files,resize_shape):
	print("Loading images from ", folder_name)
	images = []
	if image_files is None:
		image_files = os.listdir(folder_name)
	for file in image_files:
		#print("Loading image ", file, " from ", folder_name)
		image_path = folder_name + "/" + file		
		image = cv2.imread(image_path)
        
		image = cv2.resize(image,(resize_shape[1],resize_shape[0]))
		if image is not None:
			images.append(image)
		else:
			print("Error in loading image ", image)

	return images




def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    image_folder_name = "./Data/Train"
    patch_folder_name = "./Data/Train_patches"
    text_file_path = "./Code/TxtFiles/LabelsTrain.txt"
    op_folder_name = "./Results"
    """
    Read a set of images for Panorama stitching
    """

    if not os.path.exists(op_folder_name):
        os.makedirs(op_folder_name)	    
    if not os.path.exists(patch_folder_name):
        os.makedirs(patch_folder_name)    
    resize_shape = [420,420]
	# FIle/Image handling functions call
    folder_names = os.listdir(image_folder_name)
    loaded_images = loadImages(image_folder_name, folder_names, resize_shape)
    
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()