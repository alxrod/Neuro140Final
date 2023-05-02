
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset, read_image, read_mask 
from train import iou

def mask_parse(mask):
    mask
    filter = np.array([[1,1/256,1/(256*256)]])
    filtered = np.matmul(filter, mask)  
    return mask

if __name__ == "__main__":
    ## Dataset
    # path = "data/AC4/"
    # images_path = os.path.join(path, "ac4_EM/*")
    # masks_path = os.path.join(path, "ac4_seg_daniel/*")
    path = "./data/AC3/"
    images_path = os.path.join(path, "ac3_EM/*")
    masks_path = os.path.join(path, "ac3_dbseg_images/*")
    batch_size = 1
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(images_path, masks_path)
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x, as_str=True)
        y = read_mask(y, as_str=True)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        print(y_pred.max())
        # h, w, _ = x.shape
        # white_line = np.ones((h, 10, 3)) * 255.0

        # all_images = [
        #     x * 255.0, white_line,
        #     mask_parse(y_pred) * 255.0
        # ]
        # image = np.concatenate(all_images, axis=1)
        # cv2.imwrite(f"results/{i}.png", mask_parse(image))
