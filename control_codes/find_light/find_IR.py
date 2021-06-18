# import the necessary packages
from imutils import contours # pip3 install --upgrade imutils
from skimage import measure # pip3 install --upgrade scikit-image
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import argparse
import imutils
import cv2
import math

neighborhood = 50

def find_loc(image):
	centers = []
	thresh = image
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray, (11, 11), 0)


	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)
        #cv2.imshow('Threshold',thresh)
	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large"
	# components
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > image.shape[0]/2.5:
			mask = cv2.add(mask, labelMask)


	# find the contours in the mask, then sort them from left to
	# right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#print(cnts)
	cnts = contours.sort_contours(cnts)[0]
	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the bright spot on the image
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)

		#if radius<50:
		#	continue
		#print(radius)
		#print(i)
		#print([cX, cY, radius])
		cX,cY = int(cX-image.shape[1]/2), int(image.shape[0]/2 - cY)
		print('cx,cy',cX,cY)
		#if ((math.sqrt(((cX)**2) + ((cY)**2))) <= 0.9*(image.shape[0])/2):
		centers.append([cX, cY])

		#cX, cY = int(cX-img.shape[1]/2), int(img.shape[0]/2 - cY)
		cv2.circle(image, (int(cX), int(cY)), int(radius),
			(0, 0, 255), 3)
                
	return centers, image

# preprocessing
def pre_process(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (width,height))
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)[1]
    return img


def find_nearest_loc(bkg, img, x=0,y=0):
	img = cv2.subtract(img, bkg)                    # Updated by moh
	img = pre_process(img)          # Process after subtraction

	#print(img.shape)
	plt.imshow(img)
	plt.show()

	centers,img = find_loc(img)
	# Write centers in the shared CSV file

def main():
	height,width = 480,640
	bkg = cv2.resize(cv2.imread('BKGA1.png',0), (width, height), interpolation = cv2.INTER_AREA)
	img = cv2.resize(cv2.imread('IMGA1.png',0), (width, height), interpolation = cv2.INTER_AREA)
	T = find_nearest_loc(bkg, img, x=0,y=0)
	print(T)


if __name__ == '__main__':
	main()