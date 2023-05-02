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
from skimage.io import imread
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01
from tqdm import tqdm

from mem_dataset import SegmentationDataSet 
from unet import UNet 

def calculate_accuracy(output, target):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    correct = (output == target).float().sum()
    return correct / (target.numel())

def calculate_iou(output, target):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection

    iou = intersection / (union + 1e-6)  # Adding a small value to avoid division by zero
    return iou


if __name__ == "__main__":

  raw_data_path = "data/raw/"
  raw_data_paths = [raw_data_path+em for em in sorted(os.listdir(raw_data_path))]

  labeled_data_path = "data/bin_true_membranes/"
  labeled_data_paths = [labeled_data_path+em for em in sorted(os.listdir(labeled_data_path))]
  labeled_data_paths.reverse()

  pairs = {"raw": raw_data_paths, "label": labeled_data_paths}
  pairs_df = pd.DataFrame(pairs)

  pairs_df.to_csv("data/membrane_dataset.csv")
  dataset_len = pairs_df.shape[0]	

  train_df = pairs_df[0:int(0.8*dataset_len)]
  test_df = pairs_df[int(0.8*dataset_len):dataset_len]
  train_df.to_csv("data/train_membrane_dataset.csv")
  test_df.to_csv("data/train_membrane_dataset.csv")

  transforms = ComposeDouble([
    FunctionWrapperDouble(normalize_01, input=True, target=True)
  ])

  training_dataset = SegmentationDataSet(inputs=train_df["raw"],
                                       targets=train_df["label"],
                                       transform=transforms)

  test_dataset = SegmentationDataSet(inputs=test_df["raw"],
                                       targets=test_df["label"],
                                       transform=transforms)

  dataloader_train = DataLoader(dataset=training_dataset,
                                 batch_size=4,
                                 shuffle=True)

  dataloader_test = DataLoader(dataset=test_dataset,
                                   batch_size=4,
                                   shuffle=True)

  in_chan = 1
  out_chan = 1
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = UNet(in_chan, out_chan).to(device)


  checkpoint_path = 'checkpoints/model_epoch_99.pth' 
  starting_epoch = 99
  model.load_state_dict(torch.load(checkpoint_path))

  
  if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)



  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  num_epochs = 110
  for epoch in range(starting_epoch, num_epochs):
    model.train()

    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0

    for images, masks in tqdm(dataloader_train):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        outputs = outputs.to(device)
        masks = masks.to(device)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_accuracy += calculate_iou(outputs, masks).item()
        num_batches += 1

    epoch_loss /= num_batches
    epoch_accuracy /= num_batches
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Save the model checkpoint
    try:
      state_dict = model.module.state_dict()
    except AttributeError:
      state_dict = model.state_dict()

    torch.save(state_dict, f'checkpoints/model_epoch_{epoch+1}.pth')

