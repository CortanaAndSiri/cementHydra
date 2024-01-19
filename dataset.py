# encoding: utf-8
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
# from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, image_path = "data/", mode = "train"):
        assert mode in ("train", "val","test")
        self.image_path = image_path
        # self.image_list = glob(os.path.join(self.image_path, "*.bmp"))
        self.image_list = os.listdir(os.path.join(self.image_path,"images"))
        if mode == "test":
            self.image_path = image_path
            self.image_list =os.listdir(os.path.join(self.image_path,"testImages"))
            self.mask_path = os.path.join(self.image_path,"testMasks")
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = os.path.join(self.image_path,"masks")


        # self.transform_x = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485, 0.456], [0.229, 0.224])])
        self.transform_x = T.Compose([ T.ToTensor()])


    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = self.image_list[index]
            X = Image.open(os.path.join(os.path.join(self.image_path,"images"),image_name))
            
            # mask = np.array(Image.open(os.path.join(self.mask_path, image_name+".jpg")).convert('1').resize((256, 256)))
            # masks = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
            # masks[:, :, 0] = mask
            # masks[:, :, 1] = ~mask
            # X = self.transform_x(X)
            # masks = self.transform_mask(masks) * 255

            masks = Image.open(os.path.join(self.mask_path, image_name))
            masks = torch.from_numpy(np.array(masks))

            X = self.transform_x(X)
            return X, masks.squeeze().long()
        
        else:
            image_name = self.image_list[index]
            X = Image.open(os.path.join(os.path.join(self.image_path,"testImages"),self.image_list[index]))

            masks = Image.open(os.path.join(self.mask_path, image_name))
            masks = torch.from_numpy(np.array(masks))

            X = self.transform_x(X)
            return X, masks.squeeze().long()

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    pass