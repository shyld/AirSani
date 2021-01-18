# import the necessary packages
from imutils import contours # pip3 install --upgrade imutils
from skimage import measure # pip install --upgrade scikit-image

import numpy as np
import argparse
import imutils
import cv2

def find_loc(image):
	# preprocessing:
	#print('find_loc: ',image.shape)
	w0,h0 = image.shape[0],image.shape[1]
	w1= 600
	h1 = int(w1*(w0/h0))
	#print('find_loc: ',image.shape)
	image = cv2.resize(image, (w1,h1))
	#print('find_loc: ',image.shape)
	centers = []
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)

	# threshold the image to reveal light regions in the
	# blurred image
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)

	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large"
	# components
	labels = measure.label(thresh, neighbors=8, background=0)
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
		if numPixels > 0:
			mask = cv2.add(mask, labelMask)


	# find the contours in the mask, then sort them from left to
	# right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
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
		# standardize centers:
		cx,cy = (center[0]-w1/2)*(w0/w1), (center[1]-h1/2)*(h0/h1)


		centers.append([cX, cY])

		cv2.circle(image, (int(cX), int(cY)), int(radius),
			(0, 0, 255), 3)

	image = cv2.resize(image, (w0,h0))
	return centers, image

