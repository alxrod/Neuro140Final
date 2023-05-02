import numpy as np
from scipy.ndimage import measurements
from skimage.measure import label
import math

def calcIOU(mask1, mask2, eps=1e-7):
  intersection = np.logical_and(mask1, mask2)
  intersection_count = np.sum(intersection)
  
  union = np.logical_or(mask1, mask2)
  union_count = np.sum(union)
  
  iou = (intersection_count + eps) / (union_count + eps)
  return iou

def calcCentroid(mask):
  # Get the coordinates of the instance pixels
  coords = np.argwhere(mask == 1)

  # Calculate the centroid of the instance
  y_centroid = np.mean(coords[:, 0])
  x_centroid = np.mean(coords[:, 1])

  return int(x_centroid), int(y_centroid)

def calcMaxDim(mask):
  non_zero_indices = np.nonzero(mask)
  z_indices, y_indices, x_indices = non_zero_indices

  # Calculate the width and height of the instance
  width = np.max(x_indices) - np.min(x_indices)
  height = np.max(y_indices) - np.min(y_indices)

  # Return the maximum of the width and height
  return max(width, height)

def zero_outside_region(array, top_left, top_right, bottom_left, bottom_right):
  # Create a new array filled with zeros with the same shape as the input array
  new_array = np.zeros_like(array)

  # Copy the region specified by the indices from the input array to the new array
  new_array[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1] = array[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1]

  return new_array

def calcEntireIOU(prediction, ground_truth, threshold=0.5):
  pred_binary_mask = np.where((prediction > threshold), 1, 0)
  bodies_only = np.where((ground_truth > threshold), 1, 0)

  return calcIOU(pred_binary_mask, bodies_only)

def calc_batch_acc(prediction, ground_truth, threshold=0.5, eps=1e-7):
  # Threshold the prediction to create a binary mask
  pred_binary_mask = np.where((prediction > threshold), 1, 0)

  bodies_only = np.where((ground_truth > threshold), 1, 0)

  labeled_gt = label(bodies_only)   
  labeled_pred = label(pred_binary_mask)


  print("Total ", len(np.unique(labeled_gt)), " and predicted ",len(np.unique(labeled_pred)))

    
  
  avg_iou = 0
  missed = 0
  for val in np.unique(labeled_gt):
    mask = (labeled_gt == val) 
    true_x, true_y = calcCentroid(mask)
    maxDim = calcMaxDim(mask)
    end = 1024-1
    
    #reduced_pred = zero_outside_region(pred_binary_mask, 
    #  [max(true_x - 2*maxDim, 0), max(true_y - 2*maxDim, 0)], #TL
    #  [min(true_x + 2*maxDim, end), max(true_y - 2*maxDim, 0)], #TR
    #  [max(true_x - 2*maxDim, 0), min(true_y + 2*maxDim, end)], #BL
    #  [min(true_x + 2*maxDim, end), min(true_y + 2*maxDim, end)]
    #)

    found_pred = False

    min_delta = pow(maxDim,2)
    min_pred = labeled_pred
    for val in np.unique(labeled_pred):
      if val == 0:
        continue

      pred = (labeled_pred == val)
      x, y = calcCentroid(pred)
  
      cur_delta = math.sqrt(pow(true_x - x,2) + pow(true_y - y, 2))
      if cur_delta < min_delta:
        found_pred = True
        min_delta = cur_delta
        min_pred = pred
    
    if not found_pred:
      missed += 1
    else:
      avg_iou += calcIOU(mask, min_pred)

  avg_iou /= (len(np.unique(labeled_gt))-missed+eps)        
  missed_perc = float(missed) / (len(np.unique(labeled_gt))+eps) 

  print("IOU ", avg_iou, " MISSED: ", missed_perc)

  return missed_perc, avg_iou 

def get_instance_masks(instance_center, offset_x, offset_y, threshold=0.5, min_size=10):
    # Apply sigmoid function and threshold to obtain binary instance centers
    instance_center = torch.sigmoid(instance_center).squeeze().cpu().numpy()
    instance_center = (instance_center > threshold).astype(np.uint8)

    # Get the offsets
    offset_x = offset_x.squeeze().cpu().numpy()
    offset_y = offset_y.squeeze().cpu().numpy()

    # Find connected components in the instance_center mask
    labeled_instance_centers, num_instances = measurements.label(instance_center)

    instance_masks = []
    for i in range(1, num_instances + 1):
        # Extract the current instance center
        instance_center_i = (labeled_instance_centers == i)

        # Calculate the mean offsets for the current instance
        mean_offset_x = np.mean(offset_x[instance_center_i])
        mean_offset_y = np.mean(offset_y[instance_center_i])

        # Shift the instance_center_i mask according to the mean offsets
        shifted_mask_i = np.roll(instance_center_i.astype(np.float32), 
                                 shift=(int(round(mean_offset_y)), int(round(mean_offset_x))), 
                                 axis=(0, 1))

        # Remove small instances
        if np.sum(shifted_mask_i) >= min_size:
            instance_masks.append(shifted_mask_i)

    return instance_masks
