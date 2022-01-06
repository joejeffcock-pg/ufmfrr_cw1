import numpy as np
import cv2
from scipy import ndimage # multidimensional image processing
from skimage.color import rgb2hsv # image processing algorithms

def Conventional(image,Lower,Upper):
  img = image

  #Extract Red From Image
  bgrLower = Lower # minimum color(BGR)
  bgrUpper = Upper # maximum color(BGR)
  img_mask = cv2.inRange(img, bgrLower, bgrUpper) # make mask
  result = cv2.bitwise_and(img, img, mask=img_mask) # combine mask

  #Convert to greyscale
  gimg = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

  #Apply threshold
  thresh, bw = cv2.threshold(gimg, thresh=90, maxval=255, type=cv2.THRESH_BINARY_INV)
  bw = np.invert(bw)

  #Erode and Dilate until solid blocks
  #dilation
  kernel = np.ones((5,5),np.uint8)
  dilation = cv2.dilate(bw, kernel, iterations = 10)

  # closing
  kernel = np.ones((3,3),np.uint8)
  closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=4)

  # erosion
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))
  erosion = cv2.erode(closing,kernel,iterations = 5)

  # closing
  kernel = np.ones((11,11),np.uint8)
  closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=5)

  # erosion
  kernel = np.ones((7,7),np.uint8)
  erosion2 = cv2.erode(closing,kernel,iterations = 2)

  # remove boarder pixels
  erosion2[:50, :] = 0
  erosion2[:, :50] = 0
  erosion2[-50:, :] = 0
  erosion2[:, -50:] = 0

  #Count and Display
  display = img.copy()
  labels, nlabels = ndimage.label(erosion2)  # Label features in an array. Any non-zero values in input are counted as features and zero values are considered the background.
  print("There are " + str(nlabels) + " apples")

  centroid = ndimage.center_of_mass(erosion2, labels, np.arange(nlabels) + 1 ) # calculate the center of mass of the values of an array at labels.
  return(centroid)
