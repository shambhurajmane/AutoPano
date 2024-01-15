import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb

# Load the images
img1 = cv2.imread('./Data/Train/Set1/1.jpg')
img2 = cv2.imread('./Data/Train/Set1/2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize AKAZE detector
akaze = cv2.AKAZE_create()

# Detect and compute features
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)
ipdb.set_trace()
# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
ipdb.set_trace()
matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# Display the image with matplotlib
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.show()