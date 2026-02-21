import torch
import torch.nn as pd
import torch.nn.functional as F

class SimpleCNN(pd.Module):
  """
  A simple Convolutional Neural Network for image classification.
  """
  def __init__(self, num_classes: int = 6):
    super().__init__()
    
    self.conv1 = pd.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.pool1 = pd.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv2 = pd.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.pool2 = pd.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv3 = pd.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool3 = pd.MaxPool2d(kernel_size=2, stride=2)
    
    self.fc1 = pd.Linear(128 * 18 * 18, 512)
    self.dropout = pd.Dropout(0.5)
    self.fc2 = pd.Linear(512, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    
    x = torch.flatten(x, 1)
    
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    
    return x
