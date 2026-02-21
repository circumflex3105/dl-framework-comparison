import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.data_loader import create_dataset_dataframe, TRAIN_DIR, TEST_DIR
from src.pytorch.dataset import IntelImageDataset
from src.pytorch.model import SimpleCNN

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_model(model, train_loader, criterion, optimizer, device):
  """
  Trains the model for one epoch.
  """
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0
  
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device):
  """
  Evaluates the model on the test set.
  """
  model.eval()
  running_loss = 0.0
  all_preds = []
  all_labels = []
  
  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      
      running_loss += loss.item() * inputs.size(0)
      _, predicted = torch.max(outputs.data, 1)
      
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      
  epoch_loss = running_loss / len(test_loader.dataset)
  return epoch_loss, all_labels, all_preds

def main():
  """
  Main function to train and evaluate the PyTorch CNN.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  
  train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  print("Loading dataframes...")
  train_df = create_dataset_dataframe(str(TRAIN_DIR))
  test_df = create_dataset_dataframe(str(TEST_DIR))
  
  print("Creating datasets and dataloaders...")
  train_dataset = IntelImageDataset(train_df, transform=train_transform)
  test_dataset = IntelImageDataset(test_df, transform=test_transform)
  
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
  
  model = SimpleCNN(num_classes=len(train_dataset.classes)).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
  print("\nStarting training...")
  start_time = time.time()
  
  for epoch in range(EPOCHS):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
  training_time = time.time() - start_time
  print(f"\nTraining completed in {training_time:.2f} seconds.")
  
  print("\nEvaluating on test set...")
  start_time = time.time()
  test_loss, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
  inference_time = time.time() - start_time
  
  accuracy = accuracy_score(y_true, y_pred)
  print(f"Inference completed in {inference_time:.2f} seconds.")
  print(f"\nPyTorch CNN Accuracy: {accuracy * 100:.2f}%")
  
  print("\nClassification Report:")
  print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

if __name__ == "__main__":
  main()
