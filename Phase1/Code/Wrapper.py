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

import ipdb
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import rotate
import os
import sklearn.cluster
from scipy import signal, ndimage
 
# Add any python libraries here
op_folder_name = "./Results"

# To load images from given folder
def loadImages(folder_name, image_files):
	print("Loading images from ", folder_name)
	images = []
	if image_files is None:
		image_files = os.listdir(folder_name)
	for file in image_files:
		print("Loading image ", file, " from ", folder_name)
		image_path = folder_name + "/" + file
		
		image = cv2.imread(image_path)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading image ", image)

	return images


def maxpooling(given_image, pool_size):
	image_size_x = given_image.shape[0]
	image_size_y = given_image.shape[1]

	maxpool_image = np.zeros((int(image_size_x/pool_size),int(image_size_y/pool_size)))
	for i in range(maxpool_image.shape[0]):
		for j in range(maxpool_image.shape[1]):
			max = 0
			start_x = i * pool_size
			start_y = j * pool_size
			for shift_x in range(pool_size):
				for shift_y in range(pool_size):
					# print(given_image[start_x+shift_x, start_y+shift_y])
					if max < given_image[start_x+shift_x, start_y+shift_y]:
						max = given_image[start_x-shift_x, start_y-shift_y]
			maxpool_image[i][j] = max
	return maxpool_image

def avgpooling(given_image, pool_size):
	image_size_x = given_image.shape[0]
	image_size_y = given_image.shape[1]

	avgpool_image = np.zeros((int(image_size_x/pool_size),int(image_size_y/pool_size)))
	for i in range(avgpool_image.shape[0]):
		for j in range(avgpool_image.shape[1]):
			avg = 0
			start_x = i * pool_size
			start_y = j * pool_size
			for shift_x in range(pool_size):
				for shift_y in range(pool_size):
					avg += given_image[start_x+shift_x, start_y+shift_y]	
			avgpool_image[i][j] = avg/(pool_size*pool_size)
	return avgpool_image

def bilinearpooling(given_image):
	image_size_x = given_image.shape[0]
	image_size_y = given_image.shape[1]

	bilinearpool_image = np.zeros((int(image_size_x/2),int(image_size_y/2)))
	for i in range(bilinearpool_image.shape[0]):
		for j in range(bilinearpool_image.shape[1]):
			avg = 0
			start_x = i * 2
			start_y = j * 2
			for shift_x in [-1,1]:
				for shift_y in [-1,1]:
					avg += given_image[start_x+shift_x, start_y+shift_y]	
			bilinearpool_image[i][j] = avg/(2*2)
	return bilinearpool_image

def create_visualization(images, Labels, cols, size,file_name):
	rows = int(np.ceil(len(images)/cols))
	plt.subplots(rows, cols, figsize=size)
	for index in range(len(images)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(images[index], cmap= "gray")
		plt.title(Labels[index])	
	plt.savefig(file_name)
	plt.close()
	plt.show()

def create_image(image,file_name):
	plt.imshow(image, cmap='gray')
	plt.savefig(file_name)
	plt.close()
	plt.show()

def standardize_with_given_variance(image, mean, variance):
	standardized_image = np.zeros(image.shape)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			standardized_image[i][j] = (image[i][j] - mean)/variance
	return standardized_image

def feature_descriptor(image, corner):
	# feature_dict = {}
	descriptor = []
	KeyPoints = []	
	visualize = []
	for i in range(len(corner)):
		x, y = corner[i][0][0], corner[i][0][1]
		if x-21 < 0 or y-21 < 0 or x+21 > image.shape[0] or y+21 > image.shape[1]:
			continue
		patch = image[x-20:x+21, y-20:y+21]
		
		# cv2.circle(patch,(x,y),10,255,-1)
		# cv2.circle(image,(y,x),10,255,-1)

		# patch = cv2.GaussianBlur(patch,(5,5),0)
		# patch = cv2.resize(patch, (8,8), interpolation = cv2.INTER_AREA)
		# Convert to float for more precision
		patch = np.float32(patch)

		# Standardize the image to have mean=0 and variance=1
		mean, std_dev = cv2.meanStdDev(patch)
		mean = mean[0][0]
		std_dev = std_dev[0][0] if std_dev[0][0] > 0.0 else 1.0
		patch = cv2.subtract(patch, mean)
		patch = cv2.divide(patch, std_dev)

		# Convert back to uint8 if necessary
		patch = np.uint8(patch)

		patch = patch.flatten()
		descriptor.append(patch)
		KeyPoints.append((y,x))

	# create_visualization(visualize, ["patch"+str(i) for i in range(len(visualize))], 5, (10, 10),op_folder_name+"/"+"feature_descriptor.png")
	# print("feature_dict", len(feature_dict))
	# ipdb.set_trace()	
	return KeyPoints, np.array(descriptor)


def mataching_features( descriptors1, descriptors2 ):
	match = []
	thresh = 0.9

	for i in range(len(descriptors1)):
		key_dist_pair = {}
		for j in range(len(descriptors2)):
			# dist = np.linalg.norm(descriptors1[i] - descriptors2[j], ord=2)
			dist = np.sum((descriptors1[i] - descriptors2[j])**2)
			key_dist_pair[j] = dist
		best_match_key1 = min(key_dist_pair, key=key_dist_pair.get)	
		best_match_dist1 = key_dist_pair.pop(best_match_key1)
		# ipdb.set_trace()

		if best_match_dist1 > 100000:
			continue

		best_match_key2 = min(key_dist_pair, key=key_dist_pair.get)
		best_match_dist2 = key_dist_pair.pop(best_match_key2)

		best_match_ratio = best_match_dist1/best_match_dist2
		# ipdb.set_trace()
		# ratio test : This test checks if one descriptor has a unique match with descriptor in other image if not then it is not a good match. 
		if best_match_ratio < thresh:
			dmatch1 = cv2.DMatch(_queryIdx=i, _trainIdx=best_match_key1, _distance=np.float32(best_match_dist1))
			# ipdb.set_trace()

			match.append(dmatch1)
	return match
	

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures


    # Read a set of images for Panorama stitching
  
	# Set directories 
	image_folder_name = "./Data/Train/Set1"
	op_folder_name = "./Results"
	if not os.path.exists(op_folder_name):
		os.makedirs(op_folder_name)	


	# FIle/Image handling functions call
	folder_names = os.listdir(image_folder_name)
	loaded_images = loadImages(image_folder_name, folder_names)
	create_visualization(loaded_images, folder_names, 3, (10, 10),op_folder_name+"/"+"Input_images.png")

	# Corner Detection
	# Save Corner detection output as corners.png

	# Find Harris corners
	list_of_descriptors = []
	list_of_keypoints = []
	for j, img in enumerate(loaded_images):
		temp = img.copy()	
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_img = np.float32(gray_img)
		corners = cv2.goodFeaturesToTrack(gray_img,1000,0.01,10)
		corners = np.int0(corners)
		for i in corners:
			x,y = i.ravel()
			cv2.circle(temp,(x,y),3,255,-1)
		create_image(temp,op_folder_name+"/"+"corners"+str(j)+".png")
		keypoints, descriptor = feature_descriptor(gray_img, corners)	
		list_of_descriptors.append(descriptor)
		list_of_keypoints.append(keypoints)	

	match  = mataching_features(list_of_descriptors[0], list_of_descriptors[2])

	keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in list(list_of_keypoints[0])]
	keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in list(list_of_keypoints[2])]

	# Initialize BFMatcher
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# Match descriptors
	# matches = bf.match(list_of_descriptors[0], list_of_descriptors[1])	

	# Sort matches by distance
	# matches = sorted(match)

	# ipdb.set_trace()
	matched_image = cv2.drawMatches(loaded_images[0], keypoints1, loaded_images[2], keypoints2, match, None, flags=2)

	create_image(matched_image, op_folder_name+"/"+"match.png")	


		
	# Assuming keypoints1 and keypoints2 are your matched keypoints
	# And match is your list of DMatch objects

	# Extract location of good matches
	points1 = np.float32([keypoints1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
	points2 = np.float32([keypoints2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

	# Compute Homography using RANSAC
	H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use mask to remove outliers
	matchesMask = mask.ravel().tolist()

	# Draw matches
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					singlePointColor = None,
					matchesMask = matchesMask, # draw only inliers
					flags = 2)

	matched_image = cv2.drawMatches(loaded_images[0], keypoints1, loaded_images[2], keypoints2, match, None, **draw_params)

	create_image(matched_image, op_folder_name+"/"+"Rmatch.png")





	
	# create_visualization(visualize, Labeled, 2, (10, 10),op_folder_name+"/"+"corners.png")	

# """
# Perform ANMS: Adaptive Non-Maximal Suppression
# Save ANMS output as anms.png
# """
	
		
			



"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""


"""
Feature Matching
Save Feature Matching output as matching.png
"""

"""
Refine: RANSAC, Estimate Homography
"""

"""
Image Warping + Blending
Save Panorama output as mypano.png
"""


if __name__ == "__main__":
    main()
