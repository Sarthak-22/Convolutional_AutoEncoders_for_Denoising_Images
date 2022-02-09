import torch
import sys
from torchvision import transforms
from PIL import Image
import numpy as np
from model import Conv_AE


if (len(sys.argv>2)):
    try:
        img_path = sys.argv[2]
    except:
        assert('Enter correct file path')
        exit
  
  
# Pre-processing the input image  
def pre_processing(img_path):
    image = np.array(Image.open(img_path), dtype=np.float32)/255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((420,540)),
        transforms.ToTensor(),
    ])

    processed_img = transform(image)


# Passing the procesed input to the trained model
model = Conv_AE()
model.load_state_dict(torch.load())
