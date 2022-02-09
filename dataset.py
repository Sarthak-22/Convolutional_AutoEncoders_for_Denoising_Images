from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((420,540)),
                transforms.ToTensor(),
            ])


class NoisyDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=transform):
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

        if self.transform is not None:
            pre_processing = self.transform
            image, label = pre_processing(image), pre_processing(label)

        return (image,label)




def test():
    test_dataset = NoisyDataset(image_dir='dataset/train', label_dir='dataset/train_cleaned', transform=transform)
    test_image, test_label = test_dataset[0][0], test_dataset[0][1]         # Index = 0     
    print(test_image.max(), test_image.min(), test_label.max(), test_label.min())


    plt.figure()

    plt.imshow((test_image*255.0).numpy().transpose(1,2,0).astype(np.uint8), cmap='gray')
    plt.show()
    
    plt.imshow((test_label*255.0).numpy().transpose(1,2,0).astype(np.uint8), cmap='gray')
    plt.show()




#test()