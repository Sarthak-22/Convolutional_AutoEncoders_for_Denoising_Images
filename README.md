# Convolutional-AutoEncoders-for-Denoising-Images

## Architecture 
* The architecture for **Convolutional AutoEncoder** is implemented in **PyTorch**. 
* It uses 2 convolution layers (downsampling) for the **encoder** and 2 transposed convolutional layers (upsampling) for the **decoder**.
* Each Convolution layer is followed by a ReLU operation except for the last layer which is followed by a sigmoid activation unit.
* The input to the model is a torch Tensor of shape [1, 1, 420, 540]. The output is of the same shape with normalized pixel values.


## Dataset
* The dataset can be downloaded from [here](https://www.kaggle.com/c/denoising-dirty-documents/data)
* The dataset has 3 folders - 
`train`
`train_cleaned`
`test`
* The _train_ and _train_cleaned_ folders contain the input and ground truth denoised image. Test images (without GT) are present in _test_ folder
Example _train_ and _train_cleaned_ images - 

### Train Noisy Input  
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/train_noisy.png) 

### Train Denoised GT
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/train_denoised_GT.png)

## Training Details
* The model was trained for __64 epochs__ with a __batch size of 16__ and __Learning rate = 0.001__
* __Mean Squared Error (MSE) loss__ was used as a loss metric.
* Training Progress - 
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/training_loss.svg)

## Results
* Inference done on training data - 

### Train Noisy Input
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/train_noisy.png)

### Reconstructed Denoised Image
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/train_reconstructed_denoised.png)

### Train Denoised GT
![image](https://github.com/Sarthak-22/Convolutional_AutoEncoders_for_Denoising_Images/blob/main/images/train_denoised_GT.png)





