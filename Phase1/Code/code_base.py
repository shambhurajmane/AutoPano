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


def Compute_H_matrix(p_list, p_dash_list,is_SVD=False):
	"""	
	This function calculates Homography matrix using 2 methods solving linear equations and using SVD.
	This function calculates Homography matrix from first 4 points from 2 given lists.
	input:
	p_lis, p_dash_list : python lists of input points 	
	p_dash = H_matrix * p
	each value in p_list and P_dash_list is a tuple(x,y)
	for reference please check: https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
	
	output: H is Homogenous matrix in np array format
	"""
	A_matrix_list = []
	
	for point_index in(range(len(p_list))): 
		
		row_1 = [p_list[point_index][0],p_list[point_index][1], 1,  
				 0,0,0, 
				(-p_dash_list[point_index][0] * p_list[point_index][0]) , (-p_dash_list[point_index][0] * p_list[point_index][1]),-p_dash_list[point_index][0]]

		row_2 = [0,0,0, 
				 p_list[point_index][0],p_list[point_index][1], 1,
		 		(-p_dash_list[point_index][1] * p_list[point_index][0]) , (-p_dash_list[point_index][1] * p_list[point_index][1]),-p_dash_list[point_index][1]]

		
		A_matrix_list.append(row_1)
		A_matrix_list.append(row_2)
	#A_matrix_list.append([0,0,0, 0,0,0, 0,0,1])	
	A_matrix = np.matrix(A_matrix_list)
	#b = b = [0]*8 + [1]
	#H = np.reshape(np.linalg.solve(A_matrix, b), (3,3))
	
	_, _, V = np.linalg.svd(A_matrix)
	H = V[-1, :]
	H = H.reshape((3,3))
	H = H/ H[2,2]

	return H


def compute_inliers(p_list,p_dash_list, H_matrix, ssd_threshold):
	
	keypoint_list = []	
	for i in range(len(p_list)):
		point = p_list[i].copy() 		
		point.append(1)
		point_np = np.array(point)
		point_np = point_np.reshape(3,1)
		#print(point_np.shape)
		#print(H_matrix.shape)
		predicted_point = np.dot(H_matrix, point_np)	
		#print(predicted_point)
		#print(p_dash_list[i])

		#p1 = np.transpose(np.matrix([p_list[i][0], p_list[i][1], 1]))
		#predicted_point = np.dot(H_matrix, p1)
		#p2 = np.transpose(np.matrix([p_dash_list[i][0], p_dash_list[i][1], 1]))
		#ssd = np.linalg.norm((predicted_point - p2))

		ssd = np.sqrt( (p_dash_list[i][0] - predicted_point[0])**2 + (p_dash_list[i][1] - predicted_point[1])**2)		
		#print(p_list[i])
		#print(p_dash_list[i])
		#print(predicted_point)
		#print(ssd)
		#print("*****")
		if ssd <= ssd_threshold:
			keypoint_list.append({'src_point' :p_list[i], 'dst_point': p_dash_list[i],'is_inlier': 1})
		else:
			keypoint_list.append({'src_point' :p_list[i], 'dst_point': p_dash_list[i],'is_inlier': 0})
			
	return keypoint_list

def RANSAC_Homography(src_point_list, dst_point_list, maxIters, ssd_threshold):
	"""
	This funciton calculates Homography matrix using RANSAC method.
	Input: 2 lists of descriptor points of 2 images
	src_point_list : Coordinates of the points in the source image.
	dst_point_list: Coordinates of the corresponding points in the destination image.
	maxIters: Maximum number of RANSAC iterations. 

	Output: 
	Homography matrix
	mask: Optional output mask set by the RANSAC algorithm to indicate inliers and outliers.	
	"""
	max_inliers_list = []	
	for iteration in range(maxIters):		
		#print(src_point_list)
		#print(dst_point_list)
		index_list = random.sample(range(0,len(src_point_list)),4)		
		rand_src_points = [src_point_list[i] for i in index_list]
		rand_dst_points = [dst_point_list[i] for i in index_list] 

		H_matrix = Compute_H_matrix(rand_src_points,rand_dst_points,is_SVD=False)

		key_point_list = compute_inliers(src_point_list,dst_point_list, H_matrix, ssd_threshold )
				
		if len([d  for d in max_inliers_list if d['is_inlier'] == 1]) <= len([d  for d in key_point_list if d['is_inlier'] == 1]):
			max_inliers_list = key_point_list
			final_h = H_matrix
	
	
	return final_h, max_inliers_list