# Convolutional-AutoEncoders-for-Denoising-Images

## Architecture 
* The architecture for **Convolutional AutoEncoder** is implemented in **PyTorch**. 
* It uses 2 convolution layers (downsampling) for the **encoder** and 2 transposed convolutional layers (upsampling) for the **decoder**.
* Each Convolution layer is followed by a ReLU operation except for the last layer which is followed by a sigmoid activation unit.
* The input to the model is a torch Tensor of shape [1, 1, 420, 540]. The output is of the same shape with normalized pixel values.


## Dataset
* The dataset can be downloaded from (here)[https://www.kaggle.com/c/denoising-dirty-documents/data]

* The dataset has 3 folders - 
`train`
`train_cleaned`
`test`
* The __train__ and __train_cleaned__ folders contain the input and ground truth denoised image. Test images (without GT) are present in __test__ folder

