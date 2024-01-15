import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Wrapper import *

op_folder_name = "./Results"


def corner_detection(loaded_images):
	visualize =[]
	Labeled =[]
	for j, img in enumerate(loaded_images):
		temp = img.copy()	
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_img = np.float32(gray_img)
		harris_corners = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
		corners = cv2.goodFeaturesToTrack(gray_img,500,0.01,10)
		corners = np.int0(corners)
		for i in corners:
			x,y = i.ravel()
			cv2.circle(temp,(x,y),3,255,-1)
		visualize.append(temp)
		create_image(temp,op_folder_name+"/"+"corners"+str(j)+".png")
		Labeled.append("Good features to track")

		# Display Harris corners
		img[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]  # Mark corners in red

		# Display the image with Harris corners
		visualize.append(img)
		Labeled.append("Harris corners")
	
	create_visualization(visualize, Labeled, 2, (10, 10),op_folder_name+"/"+"corners.png")	



# for analyzing the effect of pooling on images of different sizes and different types
def pool_analysis(loaded_images):
	for image in loaded_images:
		visualize =[]
		Labeled =[]
		grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		resize_image = cv2.resize(grayscale_image, (int(grayscale_image.shape[1]/2), int(grayscale_image.shape[0]/2)))
		visualize.append(resize_image)
		Labeled.append("cv2.resize by 2"+str(resize_image.shape))	
		resize_image = cv2.resize(grayscale_image, (int(grayscale_image.shape[1]/4), int(grayscale_image.shape[0]/4)))
		visualize.append(resize_image)
		Labeled.append("cv2.resize by 4"+str(resize_image.shape))
		resize_image = cv2.resize(grayscale_image, (int(grayscale_image.shape[1]/8), int(grayscale_image.shape[0]/8)))
		visualize.append(resize_image)
		Labeled.append("cv2.resize by 8"+str(resize_image.shape))

		# Maxpooling
		maxpool_image = maxpooling(grayscale_image, 2)
		visualize.append(maxpool_image)
		Labeled.append("maxpool by 2" + str(maxpool_image.shape))
		maxpool_image1 = maxpooling(grayscale_image, 4)
		visualize.append(maxpool_image1)
		Labeled.append("maxpool by 4" + str(maxpool_image1.shape))

		maxpool_image2 = maxpooling(grayscale_image, 8)
		visualize.append(maxpool_image2)
		Labeled.append("maxpool by 8" + str(maxpool_image2.shape))

		# Avgpooling
		avgpool_image = avgpooling(grayscale_image, 2)
		visualize.append(avgpool_image)
		Labeled.append("avgpool by 2" + str(avgpool_image.shape))
		
		avgpool_image1 = avgpooling(grayscale_image, 4)
		visualize.append(avgpool_image1)
		Labeled.append("avgpool by 4" + str(avgpool_image1.shape))
		avgpool_image2 = avgpooling(grayscale_image, 8)
		visualize.append(avgpool_image2)
		Labeled.append("avgpool by 8" + str(avgpool_image2.shape))

		# Bilinearpooling
		bilinear_image = bilinearpooling(grayscale_image)
		visualize.append(bilinear_image)
		Labeled.append("Bilinearpool by 2" + str(bilinear_image.shape))
		
		bilinear_image1 = bilinearpooling(bilinear_image)
		visualize.append(bilinear_image1)
		Labeled.append("Bilinearpool by 4" + str(bilinear_image1.shape))
		bilinear_image2 = bilinearpooling(bilinear_image1)
		visualize.append(bilinear_image2)
		Labeled.append("Bilinearpool by 8" + str(bilinear_image2.shape))

		create_visualization(visualize,Labeled, 3, (10, 10),op_folder_name+"/"+"pooling.png")
		
def nms(loaded_images):
	for j, img in enumerate(loaded_images):
		temp = img.copy()	
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_img = np.float32(gray_img)
		corners = cv2.goodFeaturesToTrack(gray_img,500,0.01,10)
		corners = np.int0(corners)
		corner_image = np.zeros(gray_img.shape)
		# ipdb.set_trace()	
		#nms
		window = np.ones((10,10))
		for i in range(len(corners)):
			x, y = corners[i][0][0], corners[i][0][1]
			corner_image[y,x] = 1	
		create_image(corner_image,op_folder_name+"/"+"without_anms"+str(j)+".png")

		anms_image = np.zeros(gray_img.shape)
		for m,n in zip(*corner_image.nonzero()):
			window = corner_image[m-5:m+5,n-5:n+5]
			if window.size == 0:
				continue
			maximum = np.max(window)
			ipdb.set_trace()

			# print(maximum)
			if corner_image[m,n] == maximum:
				anms_image[m,n] = 1
			
		create_image(anms_image,op_folder_name+"/"+"anms"+str(j)+".png")
		

def feature_descriptor(image, corner):
	visualize = []
	Labeled = []	
	feature_dict = {}
	for i in range(len(corner)):
		x, y = corner[i][0][0], corner[i][0][1]
		if x-20 < 0 or y-20 < 0 or x+20 > image.shape[0] or y+20 > image.shape[1]:
			continue
		patch = image[x-20:x+21, y-20:y+21]
		visualize.append(patch)
		Labeled.append("patch")
		patch = cv2.GaussianBlur(patch,(5,5),0)
		visualize.append(patch)
		Labeled.append("GaussianBlur")
		visualize.append(patch)
		Labeled.append("GaussianBlur")
		patch1 = cv2.resize(patch, (8,8))
		patch1 = standardize_with_given_variance(patch1, np.mean(patch1), 1)	
		visualize.append(patch1)
		Labeled.append("resize")
		patch2 = avgpooling(patch, 4)
		patch2 = standardize_with_given_variance(patch2, np.mean(patch1), 1)	
		visualize.append(patch2)
		Labeled.append("avgpooling")

		patch3 = bilinearpooling(patch)
		patch4 = bilinearpooling(patch3)
		patch4 = standardize_with_given_variance(patch4, np.mean(patch1), 1)	
		Labeled.append("bilinearpooling")
		visualize.append(patch4)
		patch = patch.flatten()
	
		# visualize.append(patch)

		print(patch.shape)
		feature_dict[(x,y)] = patch	
		
		feature = patch

		if len(visualize) == 30:
			break
		break
		# patch = cv2.resize(patch, (8,8))
		# patch = patch.flatten()
		# if i == 0:
		# 	feature = patch
		# else:
		# 	feature = np.vstack((feature, patch))
	create_visualization(visualize, Labeled, 3, (10, 10), "patch.png")
	return feature_dict
