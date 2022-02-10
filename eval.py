from torchvision import transforms
import matplotlib.pyplot as plt
from model import Conv_AE
from PIL import Image
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("IMAGE_PATH", help="Enter the input image path", type=str)
parser.add_argument("MODEL_WEIGHTS", help="Enter the trained model weights path", type=str)
args = parser.parse_args()



PATH = args.IMAGE_PATH
model_weights = args.MODEL_WEIGHTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
  

# Pre-processing the input image  
def pre_processing(img_path):
    image = np.array(Image.open(img_path), dtype=np.float32)/255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((420,540)),
        transforms.ToTensor(),
    ])

    processed_img = transform(image)
    return processed_img



# Passing the processed input to the trained model
model = Conv_AE().to(device)
model.load_state_dict(torch.load(model_weights))



# Evaluates the denoised image from an unseen input
def evaluate(img):

    denoised = model(img.unsqueeze(0).to(device))

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Noisy Image')
    ax1.imshow((img*255.0).detach().cpu().numpy()[0,:,:].astype(np.uint8), cmap='gray')

    ax2.set_title('Denoised Image')
    ax2.imshow((denoised*255.0).detach().cpu().numpy()[0,0,:,:].astype(np.uint8), cmap='gray')
    plt.show()


# Shows the denoised image and the original input
processed = pre_processing(PATH)
evaluate(processed)



