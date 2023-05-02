import os
import numpy as np
import cv2
from skimage.io import imread, imsave
from tqdm import tqdm
import matplotlib.pyplot as plt

def mask_parse(mask):
	f = np.array([[1],[1/256],[1/(256*256)]])
	filtered = np.matmul(f, mask)  
	return filtered

def conv_mask_to_border(mask):
	kernel = np.ones((4, 4), np.uint8)
	img_erosion = cv2.erode(mask, kernel, iterations=1)
	img_dilation = cv2.dilate(mask, kernel, iterations=1)
	img_ne = np.not_equal(img_erosion, img_dilation).astype(np.uint8)
	img_out = img_ne * img_dilation
	return img_out


all_segs = os.listdir("data/seg")
for seg in tqdm(all_segs):
	path = "data/seg/"+seg
	mask = imread(path)
		
    
	border = conv_mask_to_border(mask)
	gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
	th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

	imsave("data/bin_true_membranes/"+seg, im_th)
