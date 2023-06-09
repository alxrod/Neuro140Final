import torch
import torch
from torch.utils import data
from skimage.io import imread

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        
        # Load input and target
        x, y = imread(input_ID), imread(target_ID)
        # Adjust axes
        x = x.reshape((1,x.shape[0],x.shape[1]))
        y = y.reshape((1,y.shape[0],y.shape[1]))
        y = y.astype(float)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
