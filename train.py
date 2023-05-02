import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader

from tqdm import tqdm
from skimage.io import imsave

import os
import pandas as pd
import numpy as np

from postprocess import calcEntireIOU
from dataset import SegmentationDataSet 
from model import UNet 

# Generates csv files for the training and test datasets using AC3
raw_data_path = "data/raw/"
raw_data_paths = [raw_data_path+em for em in sorted(os.listdir(raw_data_path))]

outline_data_path = "data/bin_true_membranes/"
outline_data_paths = [outline_data_path+em for em in sorted(os.listdir(outline_data_path))]
outline_data_paths.reverse()

seg_data_path = "data/seg/"
seg_data_paths = [seg_data_path+em for em in sorted(os.listdir(seg_data_path))]
seg_data_paths.reverse()

all_data_paths = pd.DataFrame({
    "raw": raw_data_paths,
    "outline": outline_data_paths,
    "seg": seg_data_paths,
})

all_data_paths.to_csv("data/instance_dataset.csv")
dataset_len = all_data_paths.shape[0]

train_df = all_data_paths[0:int(0.8*dataset_len)]
train_df.to_csv("data/train_instance_dataset.csv")

test_df = all_data_paths[int(0.8*dataset_len):-1]
test_df.to_csv("data/test_instance_dataset.csv")

# What values BORDER and BODY should be in the ground truth segmentation mask
BORDER_VALUE = -1
BODY_VALUE = 1

# Pass the training dataframe to the custom dataset class
training_dataset = SegmentationDataSet(raw_paths=train_df["raw"],
                                      outline_paths=train_df["outline"],
                                      seg_paths=train_df["seg"],
                                      border_value=BORDER_VALUE,
                                      body_value=BODY_VALUE)

dataloader_train = DataLoader(dataset=training_dataset,
                                batch_size=4,
                                shuffle=True)

# Configure the model in and out channels (using 1 each)
in_chan = 1
out_chan = 1

# Configure device to use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_chan, out_chan).to(device)

starting_epoch = 0
# If we are starting training from a checkpoint, uncomment this and change starting_epoch appropriately
# checkpoint_path = 'seg_checkpoints/model_epoch_1.pth'
# model.load_state_dict(torch.load(checkpoint_path))

# enable data parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
  print(f'Using {torch.cuda.device_count()} GPUs')
  model = nn.DataParallel(model)

# Initialize the loss functions and optimizer
criterion_mask = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20

# Setup directory to store training predictions if interested
train_output = "train_predictions"
if not os.path.exists(train_output):
  os.makedirs(train_output) 

training_stats = {"epoch": [], "loss": [], "iou": []}

for epoch in range(starting_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    epoch_miss_perc = 0.0
    epoch_iou = 0.0
 
    batch_counter = 0
    for images, masks in tqdm(dataloader_train):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        pred_masks = model(images)
        
        loss = criterion_mask(pred_masks, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # If you want to save intermediate predictions, uncomment this
        # if batch_counter % 20 == 0:
        #   test_pred = pred_masks[0].detach().cpu().numpy()
        #   test_t = masks[0].detach().cpu().numpy()

        #   pred_binary_mask = np.where((test_pred > 0.5), 255, 0)
        #   gt_binary_mask = np.where((test_t > 0.5), 255, 0)

        #   imsave("train_predictions/pred_epoch_"+str(epoch)+"_b"+str(batch_counter)+".png",np.squeeze(pred_binary_mask))
        #   imsave("train_predictions/gt_epoch_"+str(epoch)+"_b"+str(batch_counter)+".png",np.squeeze(gt_binary_mask))
        
        batch_iou = 0.0
        for i in range(len(masks)):
          iou = calcEntireIOU(pred_masks[i].detach().cpu().numpy(),
                                        masks[i].detach().cpu().numpy())
          batch_iou += iou

        batch_iou /= len(masks)
        
        epoch_iou += batch_iou
        #len(dataloader_train)
        batch_counter+=1

    epoch_loss /= len(dataloader_train)
    epoch_iou /= len(dataloader_train)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f} IOU:{epoch_iou:.4f}")
    training_stats["loss"].append(epoch_loss)
    training_stats["iou"].append(epoch_iou)
    training_stats["epoch"].append(epoch)

    # Save models between epochs
    try:
      state_dict = model.module.state_dict()
    except AttributeError:
      state_dict = model.state_dict()
    
    torch.save(state_dict, f'seg_checkpoints/model_epoch_{epoch+1}.pth')
  
training_stat_df = pd.DataFrame(training_stats)
training_stat_df.to_csv("training_stats_body_"+str(BODY_VALUE)+"_border_"+str(BORDER_VALUE)+".csv")
