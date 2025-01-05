import json
import torch                    # Pytorch module 
from PIL import Image           # for checking images
import torchvision.transforms as transforms   # for transforming images into tensors
from model_template import ModelTemplate
from torchvision import models

class EfficientNetV2SModel(ModelTemplate):
    name = "Efficient Net V2 S"
    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_indices = json.load(open("class_indices.json"))

    def train(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        # Load the saved model

        # Recreate the model architecture
        self.model = models.efficientnet_v2_s(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, len(self.class_indices))

        # Load the saved weights
        self.model.load_state_dict(torch.load("models/efficientnet/efficientnet_v2_b0_trained_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def unload_model(self):
        self.model = None

    def predict(self, image_data):
        # Define the same transformations as used during training
        transform = transforms.Compose([
            transforms.Resize((self.IMG_WIDTH, self.IMG_HEIGHT)),
            transforms.ToTensor(),          # Convert to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Apply transformations / preprocessing
        image = transform(image_data)
        image = image.unsqueeze(0)

        # Move to the appropriate device
        image = image.to(self.device)

            # Perform inference
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        predicted_class_name = self.class_indices[str(predicted.item())]
        return predicted_class_name
    
if __name__ == '__main__':
    model = EfficientNetV2SModel()
    model.load_model()
    image = Image.open('models/resnet/TomatoEarlyBlight2.JPG')
    print(model.predict(image))