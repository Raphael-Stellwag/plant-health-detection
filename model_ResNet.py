import json
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model
from sklearn.model_selection import train_test_split # for splitting the data into training and testing
from tensorflow.keras.preprocessing.image import ImageDataGenerator # for data augmentation
from device_data_loader import DeviceDataLoader # for loading in the device (GPU if available else CPU)
import torchvision.io as tv_io
import torchvision.transforms.functional as F
from ResNet9 import ResNet9
from model_template import ModelTemplate


class ResNetModel(ModelTemplate):
    name = "ResNet Model"
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    data_dir = "./data2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    


    def train(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        self.model = torch.load("plant-disease-model-complete.pth", map_location=torch.device('cpu'))

    def unload_model(self):
        self.model = None

    def predict(self, image_data):
        preprocess_trans = transforms.Compose([
            transforms.Resize((self.IMG_WIDTH, self.IMG_HEIGHT)),
            transforms.ToTensor()
        ])
        processed_image = preprocess_trans(image_data)
        xb = processed_image.unsqueeze(0)
        yb = self.model(xb)
        _, preds  = torch.max(yb, dim=1)
        # train = ImageFolder(self.train_dir, transform=transforms.ToTensor())
        # return train.classes[preds[0].item()]

        class_indices = json.load(open("class_indices.json"))
        predicted_class_name = class_indices[str(preds[0].item())]
        return predicted_class_name
    
if __name__ == '__main__':
    model = ResNetModel()
    model.load_model()
    image = Image.open('TomatoEarlyBlight2.JPG')
    print(model.predict(image))