import torch
import torch.nn as nn

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
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3,3), stride=2, padding=1, output_padding=1),    # Nx1x420x540
            nn.Sigmoid(),
        )


    def forward(self,x):
        
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        return x

def test():
    input = torch.randn(1, 1, 420, 540)
    model = Conv_AE()
    print(model)
    output = model(input)
    print(output.shape)



#test()

