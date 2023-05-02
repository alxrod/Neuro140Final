import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.measure import label


from torch.utils import data
from torch.utils.data import DataLoader
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from PIL import Image
from skimage.io import imread
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01

from seg_dataset import SegmentationDataSet 
from instance_unet import UNet 
import os
from instance_evaluation_helpers import calc_batch_acc, calcEntireIOU

train_df = pd.read_csv("data/test_instance_dataset.csv")

transforms = ComposeDouble([
  FunctionWrapperDouble(normalize_01, input=True, target=True)
])

training_dataset = SegmentationDataSet(raw_paths=train_df["raw"],
                                      outline_paths=train_df["outline"],
                                      seg_paths=train_df["seg"],
                                      transform=transforms)

dataloader_train = DataLoader(dataset=training_dataset,
                                batch_size=4,
                                shuffle=True)
in_chan = 1
out_chan = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoints = ["seg_checkpoints/" + x for x in os.listdir("seg_checkpoints") if ".pth" in x]

final_data = {
   "checkpoint": [],
   "iou": [],
}
for checkpoint_path in checkpoints:
  model = UNet(in_chan, out_chan).to(device)
  model.load_state_dict(torch.load(checkpoint_path))

  if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)
    # Initialize the model, loss functions, and optimizer
  
  model.eval() 
  train_output = "train_predictions"
  
  epoch_iou = 0
  for images, masks in tqdm(dataloader_train):
    images = images.to(device)
    masks = masks.to(device)
    pred_masks = model(images)
  
    batch_iou = 0
    for i in range(len(masks)):
      iou = calcEntireIOU(pred_masks[i].detach().cpu().numpy(), masks[i].detach().cpu().numpy())
      batch_iou += iou
    
    batch_iou /= len(masks)

    epoch_iou += batch_iou
  epoch_iou /= len(dataloader_train)


  final_data["checkpoint"].append(checkpoint_path)
  final_data["iou"].append(epoch_iou)

  print(f"Checkpoint {checkpoint_path}, IOU: {batch_iou:.4f}")

final_df = pd.DataFrame(final_data)
final_df.to_csv("model_test_iou.csv")
