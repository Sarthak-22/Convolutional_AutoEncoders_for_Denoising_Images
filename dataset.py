from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class NoisyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index])
        image = np.array(Image.open(img_path), dtype=np.float32)/255.0
        label = np.array(Image.open(label_path), dtype=np.float32)/255.0
        # image = io.imread(img_path)/255.0
        # label = io.imread(label_path)/255.0
        
        if self.transform is not None:
            pre_processing = self.transform
            image, label = pre_processing(image), pre_processing(label)

            return (image,label)



processing = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((420,540)),
                transforms.ToTensor(),
            ])



test_dataset = NoisyDataset(image_dir='dataset/train', label_dir='dataset/train_cleaned', transform=processing)
test_image, test_label = test_dataset[0][0], test_dataset[0][1]         # Index = 0 
print(test_image.max(), test_image.min())
plt.figure()
plt.imshow((test_image*255).numpy().transpose(1,2,0).astype(np.uint16))
plt.show()
#plt.imshow(test_label.permute(1,2,0))

