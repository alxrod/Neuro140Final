import torch
import torch
from torch.utils import data
from skimage.io import imread
import numpy as np
import cv2

def compress_mask(rgb_mask, mask_value=1):
    gray = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(gray, 0, mask_value, cv2.THRESH_BINARY)
    return im_th.astype(float)

def invert_mask(rgb_mask):
    gray = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
    th, im_th = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    im_th = (im_th - 1)
    return im_th.astype(float)

def flip_outline(outline_mask, mask_value=-1):
    outline_mask = np.divide(outline_mask.astype(float), 255)
    outline_mask = np.multiply(outline_mask, mask_value)
    return outline_mask

def fill_behind(array1, array2):
    mask = (array1 == 0) & (array2 != 0)
    array1[mask] = array2[mask]
    return array1

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 raw_paths: list,
                 outline_paths: list,
                 seg_paths: list,
                 border_value=-1,
                 body_value=1,
                 ):
        self.raw_paths = raw_paths
        self.outline_paths = outline_paths
        self.seg_paths = seg_paths
        self.border_value = border_value
        self.body_value = body_value

        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.raw_paths)


    def __getitem__(self, index):
        # Load the input image and ground truth mask

        r_path = self.raw_paths[index]
        o_path = self.outline_paths[index]
        s_path = self.seg_paths[index]
        
        # Load input and target
        input, outline, seg = imread(r_path), imread(o_path), imread(s_path)

        # Only use if inverting background to -1, used for testing, bad results
        #inv_bg = invert_mask(seg) 
        border = flip_outline(outline, self.border_value)
        body = compress_mask(seg, self.body_value)
        
        #combo = fill_behind(inv_bg, fill_behind(border, body))
        combo = fill_behind(border, body)
        
        raw_input = input.reshape((1,input.shape[0],input.shape[1]))
        raw_mask = combo.reshape((1,combo.shape[0],combo.shape[1]))

        raw_input = raw_input.astype(float)
        raw_mask = raw_mask.astype(float)


        # Apply any necessary data augmentation and normalization here...

        # Convert to PyTorch tensors
        input = torch.from_numpy(raw_input).permute(0, 1, 2).float()
        mask = torch.from_numpy(raw_mask).permute(0, 1, 2).float()

        return input, mask
