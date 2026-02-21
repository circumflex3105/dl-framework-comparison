import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class IntelImageDataset(Dataset):
  """
  PyTorch Dataset for the Intel Image Classification dataset.
  """
  def __init__(self, dataframe: pd.DataFrame, transform=None):
    self.dataframe = dataframe
    self.transform = transform
    
    self.classes = sorted(self.dataframe['label'].unique())
    self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    img_path = self.dataframe.iloc[idx]['filepath']
    label_str = self.dataframe.iloc[idx]['label']
    
    image = Image.open(img_path).convert('RGB')
    label = self.class_to_idx[label_str]
    
    if self.transform:
      image = self.transform(image)
      
    return image, label
