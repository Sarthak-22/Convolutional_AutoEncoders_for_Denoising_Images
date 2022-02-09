import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
import torch.optim as optim
from dataset import NoisyDataset
from model import Conv_AE


# Hyperparameters
train_image_dir = 'dataset/train'
train_label_dir = 'dataset/train_cleaned'
test_image_dir  = 'dataset/test'
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
batch_size = 16
epochs = 64

# Load Data
train_dataset = NoisyDataset(image_dir=train_image_dir, label_dir=train_label_dir)
train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = Conv_AE().to(device=device)


# Loss and optimizers
criterion = nn.MSELoss()                            # Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train
for epoch in range(epochs):
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device=device)
        label = label.to(device=device)
        
        # forward
        denoised = model(image)
        loss = criterion(denoised, label)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent
        optimizer.step()

        print(f'Epochs:{epoch+1}, Loss:{loss}')

torch.save(model.state_dict(), 'model_weights.pth')
