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
import os
import random

## global variables

# Add any python libraries here
# To load images from given folder
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

def data_generation(image_list, patch_size, no_patches , perturbation_range, translation_values, active_region_xy , result_path, text_file_path ):
    """
    This function creates image data set for training of the CNN. 
    The data set has following structure.
    image patch 1 - Mp * Np * 3 
    image patch 2 - Mp * Np * 3 (warped image)
    labels - Hp4 - [Cb-Ca]
    patch_heightlist [a,b] - height, width of patch
    perturbation_range = 
    """
    counter = 0
    H_p4_list = []
    patch_width = patch_size[1]
    patch_height = patch_size[0]
    for index_1,img in enumerate(image_list):  
        
        for index_2 in range(no_patches):
            
            image_height,image_width, channels = img.shape                 
            
    
            x = random.randint(active_region_xy[1], image_width - active_region_xy[1])
            y = random.randint(active_region_xy[0], image_height - active_region_xy[0]) 
            
            ##create corners
            corners = [ [x, y], 
            [x + patch_width, y],
            [x + patch_width, y + patch_height],
            [x, y + patch_height ] ]     

            
            ##create patch, apply pertubation to corners, find H and H_inv, warp the img and read a patch from warped img, find Hp4 delta vector
            patch = read_patch(img, corners)        
            perturbed_corners = apply_perturbation(perturbation_range, translation_values ,corners)
            T_matrix = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), np.array(perturbed_corners, dtype=np.float32))
            H_inv = np.linalg.inv(T_matrix)        
            height,width,_ = img.shape
            output_image = cv2.warpPerspective(img, H_inv, (width,height), flags=cv2.INTER_LINEAR)  # Specify the output image dimensions (width, height)
            patch_a = patch
            patch_b = read_patch(output_image, corners)
            H_p4 = np.array(perturbed_corners) -np.array(corners)
            H_p4 = list(H_p4.flatten())
            H_p4_list.append(H_p4)
            counter = counter + 1
            cv2.imwrite(result_path + "/Original_" + str(counter)+ ".jpg" ,patch_a)
            cv2.imwrite(result_path + "/Warped_" + str(counter)+ ".jpg" ,patch_b)

        
                
    with open(text_file_path, 'w') as file:
        for i in range(len(H_p4_list)):
            file.write(f'{H_p4_list[i]}\n')
            
        


def read_patch(img, corners):
    #patch_height[height, width]
    
    patch = img[ corners[0][1]: corners[2][1] , corners[0][0] : corners[1][0]]
    #corners = [ [x, y], 
    #            [x + patch_width, y],
    #            [x + patch_width, y + patch_height],
    #            [x, y + patch_width] ]
    return patch

def apply_perturbation(perturbation, translation_values ,corners):
    """
    This function apply random perturbation on all 4 corners of an img and also adds translation on all 4 corners.
    """
    min_perturbation = perturbation[0]
    max_perturbation = perturbation[1]
    perturbed_corners = []
    for i in range(len(corners)):
        perturbed_corners.append( [corners[i][0] + random.randint(min_perturbation , max_perturbation ) + translation_values[0],
        corners[i][1] + random.randint(min_perturbation , max_perturbation) + translation_values[1] ] )
    return perturbed_corners


def test_patch():
    image_path = "./Data/Train" + "/" + "1.jpg"
    image = cv2.imread(image_path) 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height,image_width, channels = image.shape 
    patch_size = [100,200]
    perturbation_range = [-10,10]

    patch_width = patch_size[1]
    patch_height = patch_size[0]
    x = random.randint(0, image_width - patch_width)
    y = random.randint(0, image_height - patch_height) 
    
    ##create corners
    corners = [ [x, y], 
                [x + patch_width, y],
                [x + patch_width, y + patch_height],
                [x, y + patch_height ] ]   
   

   
    #print(image.shape)
    patch = read_patch(image, corners)
    print(corners)
    perturbed_corners = apply_perturbation(-50,50, corners)
    print((np.array(perturbed_corners)))
    cv2.imshow('Image', image)
    cv2.imshow('patch', patch)
    src = np.array(corners)
    print((src))
    dst = np.array(perturbed_corners)

    H_matrix = Compute_H_matrix(src, dst)
    print(H_matrix)

    H_matrix, _ = cv2.findHomography(src, dst)
    print(H_matrix)
    #Wait for a key event and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
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
    active_region_xy = [150,150]
    patch_size = [128,128]
    perturbation_range = [-32,32]
    translation_values = [5,5]   # fixed translation amount for all 4 corners
    no_patches = 10
    data_generation(loaded_images, patch_size, no_patches ,perturbation_range, translation_values,active_region_xy ,result_path = patch_folder_name, text_file_path = text_file_path)

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

def Compute_H_matrix(p_list, p_dash_list):
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

	


if __name__ == "__main__":
      
    main()
