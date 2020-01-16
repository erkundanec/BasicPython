## ============================ FileName.py file ============================= %
# Description                         :
# Input parameter                     : 
# Output parameter                    : 
# Subroutine  called                  : 
# Called by                           : 
# Reference                           : 
# Author of the code                  : Kundan Kumar
# Date of creation                    : 
# ------------------------------------------------------------------------------------------------------- %
# Modified on                         : 
# Modification details                :
# Modified By                         : Kundan Kumar
# ===================================================================== %
#   Copy righted by ECE Department, ITER,SOA, Bhubaneswar, India.
# ===================================================================
import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('myWatch.jpeg',0)
cv2.imshow('image',img)