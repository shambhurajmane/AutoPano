# AutoPano
The purpose of this repository is to stitch two or more images in order to create one seamless panorama image. The functions covered are Corner detection, Adaptive Non-Maximal Suppression, Feature Matching, RANSAC for outlier rejection,  Homography estimation and Blending.

## About The Project

This report presents image stitching pipelines using traditional computer vision and modern deep learning techniques. Phase 1 implements feature extraction, matching, RANSAC and blending to create panoramas. Phase 2 develops supervised and unsupervised convolutional neural networks (CNN) to predict homographies. Quantitative and qualitative results compare classical and learning methods. This project provides end-to-end experience in applying fundamental techniques alongside recent deep learning innovations for geometric computer vision tasks. The implemented pipelines could form the basis of real-world panoramic image and video capturing systems.

## Classical Approach 
  ###   Corner Detection
<img src="Phase1/Results/corners.png" alt="Logo" width="450" height="300"> 

  ###   Feature generation
<img src="Phase1/Results/feature_descriptor.png" alt="Logo" width="450" height="300"> 

  ###   Feature matching and RANSAC
<img src="Phase1/Results/before_ransac.png" alt="Logo" width="450" height="300"> 

  ###   Image stitching
<img src="Phase1/Results/sequence.png" alt="Logo" width="900" height="300"> 

## Other results
1 | 2 | 3
:---: | :---: | :---: 
<img src="Phase1/Results/Case2_[0, 1, 2].png" alt="Logo" width="450" height="200"> |<img src="Phase1/Results/test_Set2/4.png" alt="Logo" width="450" height="200">  | <img src="Phase1/Results/set3_4.png" alt="Logo" width="450" height="200"> 


## Deep Learning Approach 

  ###  Ground Truth vs Supervised model 
<img src="Phase2/Results/supervised/1.png" alt="Logo" width="450" height="300"> 
<img src="Phase2/Results/supervised/2.png" alt="Logo" width="450" height="300"> 
<img src="Phase2/Results/supervised/3.png" alt="Logo" width="450" height="300"> 

  ### Loss graph
<img src="Phase2/Results/supervised/loss_99model_corr_sup.png" alt="Logo" width="450" height="300"> 


  ### Ground Truth vs Supervised model vs unsupervised model
  
<img src="Phase2/Results/together/1.png" alt="Logo" width="450" height="300"> 
<img src="Phase2/Results/together/2.png" alt="Logo" width="450" height="300"> 
<img src="Phase2/Results/together/3.png" alt="Logo" width="450" height="300"> 
<img src="Phase2/Results/together/4.png" alt="Logo" width="450" height="300"> 

  ### Loss graph
<img src="Phase2/Results/together/Figure_1.png" alt="Logo" width="450" height="300"> 
