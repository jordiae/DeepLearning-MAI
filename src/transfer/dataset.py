from torch.utils.data.dataset import Dataset
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder  # CrossEntropyLoss expects class indices


class Mit67Dataset(Dataset):
    def __init__(self, path, transform, enc=None):
        self.X = []
        self.y = []
        for class_ in os.listdir(path):
            for img_path in os.listdir(os.path.join(path, class_)):
                img_full_path = os.path.join(path, class_, img_path)
                self.X.append(img_full_path)
                self.y.append(class_)
        self.data_len = len(self.X)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        if enc is None:
            self.enc = LabelEncoder()
            self.enc = self.enc.fit(self.y)
            self.y = self.enc.transform(self.y)
        else:
            self.enc = enc
            self.y = self.enc.transform(self.y)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.X[index])
        if self.transform is not None:
            image_np = np.array(img)
            augmented = self.transform(image=image_np)
            img = augmented['image']
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.data_len
