
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(images_path, masks_path, split=0.1):
    images = sorted(glob(images_path))
    # Specific for dbseg ordering
    masks = sorted(glob(masks_path))[::-1] 

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path, as_str=False):
    if not as_str:
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (1024, 1024))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask(path, as_str=False):
    if not as_str:
        path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_COLOR)

    # fits in multiple segments
    borders = conv_mask_to_border(mask)
    filter = np.array([[1],[256],[256*256]])
    filtered = np.matmul(borders, filter)  
    normed = filtered/255.0

    return normed

def conv_mask_to_border(mask):
    kernel = np.ones((4, 4), np.uint8)
    img_erosion = cv2.erode(mask, kernel, iterations=1)
    img_dilation = cv2.dilate(mask, kernel, iterations=1)
    img_ne = np.not_equal(img_erosion, img_dilation).astype(np.uint8)
    img_out = img_ne * img_dilation
    return img_out
    # Take masks where im_dilation != im_erode

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([1024, 1024, 1])
    y.set_shape([1024, 1024, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
