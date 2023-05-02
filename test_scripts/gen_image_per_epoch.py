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
                              shuffle=False)

in_chan = 1
out_chan = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoints = ["seg_checkpoints/" + x for x in os.listdir("seg_checkpoints") if ".pth" in x]

images, masks = next(iter(dataloader_train))
for checkpoint_path in tqdm(checkpoints):
  model = UNet(in_chan, out_chan).to(device)
  model.load_state_dict(torch.load(checkpoint_path))

  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # Initialize the model, loss functions, and optimizer
  
  model.eval() 

  epoch = checkpoint_path.split("_")[-1].split(".")[0]

  images = images.to(device)
  masks = masks.to(device)
  pred_masks = model(images)

  for i in range(len(masks)):
    test_pred = pred_masks[i].detach().cpu().numpy()
    test_t = masks[i].detach().cpu().numpy()

    pred_binary_mask = np.where((test_pred > 0.5), 255, 0)
    gt_binary_mask = np.where((test_t > 0.5), 255, 0)

    raw_name = training_dataset.raw_paths[i].split("/")[-1].split(".")[0]
    imsave("sample_predictions/pred_epoch_"+str(epoch)+"_"+raw_name+".png",np.squeeze(pred_binary_mask).astype(np.uint8))
    imsave("sample_predictions/gt_epoch_"+str(epoch)+"_b"+raw_name+".png",np.squeeze(gt_binary_mask).astype(np.uint8))

