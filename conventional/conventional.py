import numpy as np
import cv2
from scipy import ndimage # multidimensional image processing
from skimage.color import rgb2hsv # image processing algorithms

def conventional(image, thresholds):
  img = image

  # convert to HSV
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  #Extract colours From Image
  img_mask = np.zeros(img.shape[:2], dtype=np.uint8)
  for lower, upper in thresholds:
    bgrLower = np.array(lower) # minimum color(HSV)
    bgrUpper = np.array(upper) # maximum color(HSV)
    curr_mask = cv2.inRange(img_hsv, bgrLower, bgrUpper) # make mask
    img_mask = cv2.bitwise_or(img_mask, curr_mask) # combine mask
  result = cv2.bitwise_and(img, img, mask=img_mask) # apply mask

  #Convert to greyscale
  gimg = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

  # opening
  kernel = np.ones((2,2),np.uint8)
  opening = cv2.morphologyEx(gimg, cv2.MORPH_OPEN, kernel, iterations=2)
  # cv2.imshow('opening', opening)

  labels, nlabels = ndimage.label(opening)  # Label features in an array. Any non-zero values in input are counted as features and zero values are considered the background.
  centroid = ndimage.center_of_mass(opening, labels, np.arange(nlabels) + 1 ) # calculate the center of mass of the values of an array at labels.

  predictions = []
  for point in centroid:
      y, x = point
      predictions.append([int(x), int(y)])
  return(predictions)
