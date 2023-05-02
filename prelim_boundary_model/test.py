import torch
import torch.nn as nn
import torch.optim as optim
from skimage.io import imread
from torch.utils import data
from torch.utils.data import DataLoader
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01
from tqdm import tqdm

from dataset import SegmentationDataSet 
from unet import UNet 

def calculate_iou(output, target):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    iou = intersection / (union + 1e-6)  # Adding a small value to avoid division by zero
    return iou


if __name__ == "__main__":

  pairs_df = pd.read_csv("data/membrane_dataset.csv")
  dataset_len = pairs_df.shape[0]	

  train_df = pairs_df[0:int(0.8*dataset_len)]
  test_df = pairs_df[int(0.8*dataset_len):dataset_len]
  test_df = test_df.reset_index()
  train_df.to_csv("data/train_membrane_dataset.csv")
  test_df.to_csv("data/test_membrane_dataset.csv")

  transforms = ComposeDouble([
    FunctionWrapperDouble(normalize_01, input=True, target=True)
  ])
  
  test_dataset = SegmentationDataSet(inputs=test_df["raw"],
                                       targets=test_df["label"],
                                       transform=transforms)
  
  dataloader_test = DataLoader(dataset=test_dataset,
                                   batch_size=4,
                                   shuffle=True)

  in_chan = 1
  out_chan = 1
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = UNet(in_chan, out_chan).to(device)

  checkpoint_path = 'checkpoints/final_model_file.pth' 
  model.load_state_dict(torch.load(checkpoint_path))

  if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)

  model.eval()  # Set the model to evaluation mode
  test_iou = 0
  num_batches = 0

  output_folder = 'output_images'
  os.makedirs(output_folder, exist_ok=True)
  
  with torch.no_grad():
    for idx, (images, masks) in enumerate(tqdm(dataloader_test)):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        
        test_iou += calculate_iou(outputs, masks).item()
        num_batches += 1

        output_masks = torch.sigmoid(outputs)
        output_masks = (output_masks > 0.5).float()
        output_masks = output_masks.squeeze().cpu().numpy()
        
        for i, output_mask in enumerate(output_masks):
            img = Image.fromarray((output_mask * 255).astype('uint8'))
            img.save(os.path.join(output_folder, test_df["raw"][idx*len(output_masks)+i].split("/")[-1]))

  test_iou /= num_batches

  print(f'Test IoU: {test_iou:.4f}')

