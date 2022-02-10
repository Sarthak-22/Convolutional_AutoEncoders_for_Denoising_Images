import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import NoisyDataset
from model import Conv_AE
import matplotlib.pyplot as plt
import numpy as np


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


def show_train_results(index):
    image, label = train_dataset[index][0], train_dataset[index][1]
    model = Conv_AE().to(device)
    model.load_state_dict(torch.load("model_weights.pth"))

    out = model(image.unsqueeze(0).to(device))

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.set_title("Train Input Image")
    ax1.imshow((image*255.0).numpy()[0,:,:].astype(np.uint8), cmap='gray')

    ax2.set_title("Train Reconstructed Image")
    ax2.imshow((out*255.0).detach().cpu().numpy()[0,0,:,:].astype(np.uint8), cmap='gray')
    
    ax3.set_title("Train Ground Truth Image")
    ax3.imshow((label*255.0).numpy()[0,:,:].astype(np.uint8), cmap='gray')
    plt.show()


#show_train_results(5)