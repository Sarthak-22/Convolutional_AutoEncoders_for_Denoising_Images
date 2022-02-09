import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

class Conv_AE(nn.Module):
    def __init__(self):
        super(Conv_AE, self).__init__()
        
        # Nx1x420x540
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=2, padding=1),    # Nx16x210x270
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1),   # Nx32x105x135
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=2, padding=1, output_padding=1),   # Nx16x210x270
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )



        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same')
        # self.pool = nn.MaxPool2d(kernel_size=(2,2), padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding='same')
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3,3), padding='same')

        # self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(2,2), stride=(2,2))
        # self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=(2,2))

    def forward(self,x):
        # Encoder
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))

        # # Latent space
        # x = self.pool(x)
    
        # # Decoder
        # x = F.relu(self.deconv1(x))
        # x = F.relu(self.deconv2(x))
        # x = torch.sigmoid(self.conv3(x))

        x = self.encoder(x)
        x = self.decoder(x)

        return x

def test():
    input = torch.randn(1, 1, 420, 540)
    model = Conv_AE()
    print(model)
    output = model(input)
    print(output.shape)



#test()

