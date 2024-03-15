This project is part of a challenge about face recognition
![WhatsApp Image 2024-02-26 à 13 38 07_b027cde6](https://github.com/ghalys/Face_recognition_challenge/assets/127297865/a0585e44-75aa-43db-bcdc-8c7b236fff8b)


## Introduction
This project aimed to produce an automatic facial recognition system in which we chose to work with Python. The performance will be trained on a test set of 1200 facial images. The goal of this project is to identify if any user that is specified in the training dataset is present or not in the image, which will return “-1” if there’s an imposter, and if the user is in the image the model will return his/her identity, in total we have 80 different users to identify. This challenge aims to get over 20% in the testing.
To achieve this final project, it was possible to work with machine learning and/or deep-learning algorithms.
It is important to mention that to complete this challenge the following must be taken into account: layers, parameters, and maximum size. The model must have a maximum of 1,000,000 parameters, 10 layers, and not exceed 80 MB in size.

In this project, it is essential to apply knowledge acquired during this course. Such as: 
- Detection of Faces using Viola-Jones by Haar features.
- Deep Learning with the Convolutional Neural Network
- Image Processing

## State of the Art
To work with Deep Learning it is necessarily a Neural Network. Convolutional Neural Networks (CNNs) are good at understanding and working with images, speech, or sounds. As they go through each layer, they become better at recognizing different parts of the image, making them very effective in image recognition [1].
This Convolutional Neural Network works with three main layers: The convolutional layer, the Pooling Layer, and the fully connected (FC) layer. The convolutional layer is central to a CNN, handling most of the computation. It needs three main components: input data, a filter, and a feature map. The Pooling Layer scans a filter across the entire input, but unlike other filters, this one doesn't have any weights, this has two main types of pooling: Max pooling and Average Pooling [1].
To do the detection of faces, the Viola-Jones method was used. This method integrates support vector machines, boosting algorithms, and cascade classifiers. Its application extends to processing digital images to identify facial positions within them [2]. One of the reasons why this method was chosen is because its effectiveness is high and the laboratories had already worked with this method in the past.

To have the images in the right format before the training it is necessary to know how to Preprocess the image. For this point, these are some of the steps involved in image preprocessing: image loading, resizing so that all images can be uniformly sized, color correction for consistency, normalization of pixel values, and to facilitate processing, conversion to grayscale.

## Steps
To improve and achieve this challenge, it was necessary to follow a number of steps.
# Step 0: Increasing the training set size
![image_A1978](https://github.com/ghalys/Face_recognition_challenge/assets/127297865/58e39386-e3ad-4af2-9f23-2ff12729b6cf)
Given the restrictions of the challenge, several approaches were tested (most of which will be mentioned below). After a number of changes that were not yielding any results, it was  decided that the provided training set was not enough (even with augmentation). We noticed that for most of the ids, we only got 4 different images which seemed to not be enough for a Deep Learning model to train, especially one which has 81 classes. After reviewing the images, it was determined that most of them are celebrities which meant that more pictures of each one can be obtained from the Internet. The names of celebrities were obtained from an Image Recognize website [3] and then images were downloaded via a google module called: pygoogle_image and placed in corresponding folders. Then they were cleaned up manually from repetitions. Images with more than one face were also removed to achieve higher accuracy. We ended up with 3002 images to train with. Increasing the dataset improved the performance of the variously structured models.
# Step 1: 
A common practice for training AI models is to split the training data into training and validation sets. A common ratio for that is 4/1 respectively.
# Step 2: Preprocessing of images for training 
Each of the images is loaded together with its corresponding label and faces, or rather - the coordinates of their bounding boxes (taken from a previously provided MatLab file). The original dataset and the additional one are added together. The images are iterated over, cropped at the bounding box coordinates, resized to a consistent size (multiple options were tried, but for the final model 180x180 pixels size was used) such that each image has the same dimension which is crucial for training a CNN and normalized. We chose colored images to have more accuracy, but we were aware of the computational cost at training. At the end the label and image are added together such that they can be fed to the model for training.
# Step 2.1: Preprocessing of images for validation/testing 
Images for validation and/or testing are preprocessed in a slightly different way. Instead of taking the bounding boxes of the faces from a file, the face detection model built in the first challenge is used. This provides the bounding box and images are then preprocessed in the same way as for training.
![image](https://github.com/ghalys/Face_recognition_challenge/assets/127297865/7a024a3c-11fb-472e-9d49-6b7b34d15788)

# Step 3: Model Structure and Training 
The structure used for the models was not strictly determined but the main objective was to tweak it to fulfill the project’s requirements.  At first it seemed reasonable to try and find pretrained models to finetune with the initial dataset. For this we tried a VGG16 structure but it had too many parameters and the performance was not good enough to cover even the minimal requirements for the project. After trying that other options were searched but nothing was found that would account for the limitations. As such it was decided to move onto building a custom CNN with layers from Keras 3 [4]. Various combinations of layers and hyperparameters were explored. Changes were made to the number of layers, number of convolutional blocks, structure of convolutional blocks, the batch size, the size of the images, the number of filters in the convolutional layers, kernel sizes, optimizers, etc. Given that building a custom CNN is not a precise science, a lot of combinations were explored and tried but it is hard to describe them all. After these did not yield good results with the initial dataset it was decided to move on to a Siamese Neural Network structure, which in general require less training samples compared to a simple CNN as they rely on contrastive estimations [5]. This did not yield the desired results and we were back to CNN, but after more trial and error both the training and validation accuracies would get stuck at around 65 and would yield F1 scores of less than 20 (the minimum for the project). It was at this point that it was decided that the best way to improve the results would be to collect more images (the process for which is described above in Step 0). After the collection of images performances improved, but what really changed the result was changing the batch size (to 32) and the image size (to 180x180 pixels). The final improvement was obtained by switching from grayscale to color. This yielded an F1 of around 85% when validating with the provided python functions. 
# Step 4: Testing the Model
Before applying the model prediction on a testing image, we check first if the image has a face. if not, we already return -1 without using the model. And then for images that have two or more faces detected, we return the class associated with the highest probability given by the model. At the end, we got 85% for the validation set. 

## Conclusion
After several weeks of working on this project, we can conclude that the face detection and identification process requires a large amount of image preprocessing. As detailed in the previous section, the images had to be resized, colorized, normalized, etc.  Undoubtedly, one of the biggest changes we could see when training the models was to enlarge the dataset, manually adding 20 more images per person, which allowed us to train the model better and to have a higher level of accuracy and take into account the colored images. This led us to have a result of 85 in the validation test, however, the different models and different configurations were not enough to be able to raise this result. 



The project presented in this course was challenging since it managed to unite all the topics and practices seen during the quarter. We tested a large number of models trying to achieve the best results but the limitation of parameters was a really hard part to respect. Having a total of 5 layers and 707473 parameters, we achieved decent results in validation 

## What can be improved?
We believe that with more perspective and experience, we can better control the hyperparameters. We saw that the constraint 1M parameters was really a strong constraint because most of the models we saw online get better results but with more parameters. Increasing the number of parameters can be computationally costly but ensures we can get better accuracy.

