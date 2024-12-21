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

class ResNetModel:  
    name = "ResNet Model"

    def train(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.is_available()
        self.model = torch.load('plant-disease-model-complete.pth', map_location=device)
        self.model

    def predict(self, image_data):
        output = self.model(batched_image_gpu)
        output
        return "Prediction"