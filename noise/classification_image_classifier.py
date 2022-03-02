import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, classification_images):
        super().__init__()
        self.classification_images = classification_images
        self.num_classes = classification_images.size()[0]

    def forward(self, x):
        prediction = torch.zeros(x.size()[0], self.num_classes)
        for i in range(x.size()[0]):
            for j in range(self.num_classes):
                prediction[i][j] = torch.sum(self.classification_images[j] * x[i])
        return prediction,
    
    def get_name(self):
        return "classificatoin_image_classifier"