# filepath: /pam-project/pam-project/models/pretrained_model.py
# This file contains the architecture and methods for loading and fine-tuning a pretrained model.

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class PretrainedModel(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedModel, self).__init__()
        # Load a pretrained model (e.g., ResNet50)
        self.model = models.resnet50(pretrained=True)
        # Replace the final layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_pretrained_model(num_classes, device):
    model = PretrainedModel(num_classes)
    model.to(device)
    return model

def fine_tune_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
    ])