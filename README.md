## Project: Build a Traffic Sign Recognition Program

<p align="center">
<img src="readme_images\image_01.JPG" width="480" alt="Project 2: Cover" align = 'center'/> 
</p>

Overview
   ---
   
In this project, I will use convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out the model on images of German traffic signs that I found on the web.

   ---
The Project
   ---
The goals / steps of this project are the following:
* Load the data set:
   Downloaded German Traffic Sign Dataset consists of 3 pickle objects with training, validation and test data sets. After unpacking pickle objects, we can explore each of 3 data sets.
* Exploration, summary and visualization of the data set:
   Each of the data sets has traffic sign pictures 32 by 32 pixels with 3 RGB channels. Each sample is labeled as well.
   Number of samples in training set is 34799, in validation set - 4410 and in test set - 12630. It is apprx. ratio 100%(train)/13%(validation)/36%(test). Thefore there is enough samples to train and test the model.
    There are 43 different signs in the German Traffic Sign Dataset according to [signnames.csv](/signnames.csv). Distribution of the 43 sings in the data sets is following:
<p align="center">
<img src="readme_images\train_valid_test_dist.png" width="480" alt="Samples' distributon" />     
</p>
   As it can be seen, there are samples of every sign in all three datasets. Probably it should be enough to train a model without overfitting for some signs and underfitting for others.
    To be sure that data is labeled accordinly to [signnames.csv](signnames.csv), I took randomly 43 images for each label and compared their pictures to lables.
    
          ClassId                                           SignName
          0                               Speed limit (20km/h)
          1                               Speed limit (30km/h)
          2                               Speed limit (50km/h)
          3                               Speed limit (60km/h)
          4                               Speed limit (70km/h)
          5                               Speed limit (80km/h)
          6                        End of speed limit (80km/h)
          7                              Speed limit (100km/h)
          8                              Speed limit (120km/h)
          9                                         No passing
         10       No passing for vehicles over 3.5 metric tons
         11              Right-of-way at the next intersection
         12                                      Priority road
         13                                              Yield
         14                                               Stop
         15                                        No vehicles
         16           Vehicles over 3.5 metric tons prohibited
         17                                           No entry
         18                                    General caution
         19                        Dangerous curve to the left
         20                       Dangerous curve to the right
         21                                       Double curve
         22                                         Bumpy road
         23                                      Slippery road
         24                          Road narrows on the right
         25                                          Road work
         26                                    Traffic signals
         27                                        Pedestrians
         28                                  Children crossing
         29                                  Bicycles crossing
         30                                 Beware of ice/snow
         31                              Wild animals crossing
         32                End of all speed and passing limits
         33                                   Turn right ahead
         34                                    Turn left ahead
         35                                         Ahead only
         36                               Go straight or right
         37                                Go straight or left
         38                                         Keep right
         39                                          Keep left
         40                               Roundabout mandatory
         41                                  End of no passing
         42  End of no passing by vehicles over 3.5 metric .
<p align="center">
<img src="readme_images\signs.png" width="480" alt="Signs" />  
</p>
   All randomly selected samples for each label fit to labels in the [signnames.csv](signnames.csv).
   
* Design, train and test a model architecture
    As a basis for my model I took the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture of the convolutional neural networks.
    LeNet architecture proved itself to be effective solution for handwritten charecter recognition.
    Since input samles are 32 by 32 pixels and we expect to classify 43 traffic signs, the structure of the neural network will be:
<p align="center">
<img src="readme_images\NeuralNetworkArchitecture.JPG" width="480" alt="LeNet" />  
</p>
   In our case not 10 digits, but 43 traffic signs should be analyzed, thefore we changed number of the outputs to 43. Quite offen color information is not critical for neural networks, therefore I will convert images to grayscale.
   Chosen architecture in more details:
   
        Input layer 1. 32x32x1
        Convolution layer 1. The output shape should be 28x28x6.
        Activation 1. ReLU activation.
        Pooling layer 1. The output shape should be 14x14x6.
        Convolution layer 2. The output shape should be 10x10x16.
        Activation 2. ReLU activation.
        Pooling layer 2. The output shape should be 5x5x16.
        Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
        Activation 3. ReLU activation.
        Fully connected layer 2. This should have 84 outputs.
        Activation 4. ReLU activation.
        Fully connected layer 3. This should have 43 outputs.

*  Preprocession:
    3 Channel images will be converted to 1 channel image. Then part of the images with signs will be cut off the images and scaled to 32x32x1 image. I tried to normalize images (from -128 to 128)/128, but the accuracy just worsens after it. Therefore normalization was not used in the pipeline. 
    Pipeline for image preprocession looks like this:
<p align="center">
    <img src="readme_images\signs_transformations.JPG" width="480" alt="Sign transformation" />   
</p>
   This preprocession technique will be applied to all the images in the training, validation and test sets.
   I did several experiments with different epochs number and learning rates. As you can see on the picture, 50 epochs with the learning rate 0.001 should be optimal for the model:
 <p align="center">
    <img src="readme_images\accuracy-learning rate.png" width="480" alt="accuracy/learning rate" />   
</p>
After 50 epochs validation accuracy was 0.938 and test accuracy was 0.925. Thefore the model generalize data quite good. 
For training was used Adam optimizer (Adam: a method for stochastic optimization) and batch size 128 samples.
* Predictions of the model on new images
    10 images with traffic signs were downloaded from Internet.
<p align="center">
    <img src="readme_images\signs_online.PNG" width="480" alt="signs from internet" />
</p>
   Challenges in recognition of these images (the number in the top right corners were added just for informational perposes and won't be fed to the neural network) are:
   
*  1st sign is dirty;
*  2nd and 3d signs are simiar in shapes and have similar tilts. It's interesting whether the modules confuses these 2 signs;
*  4th sign is similar to the 1st sign in shape;
*  5th and 6th are the same sign under different weather conditions;
*  7th and 8th are the same sign with and without graffiti, because it can confuse a self-driving car. I.e. as [reported](http://blog.caranddriver.com/researchers-find-a-malicious-way-to-meddle-with-autonomous-cars/) earlier this month;
*  9th sign has different shape from all other signs and less frequent than other signs, thefore the model might be undertrained on this kind of images;
*  10th images doesn't have any sign on it, but the chosen model will still probably find a sign on it.
   Let's apply the preprocessing pipeline to the images and after it the neural network.
   Test accuracy is 0.900. Taking in account, that the image number 10 has no signs on it, the model performed (fortunately) with accuracy 100%, if there wouldn't be the image with noise.
<p align="center">
   <img src="readme_images\predictions.PNG" width="480" alt="top3 predictions" /> 
</p>

Softmax probabilities for 10 images:
*  picture number 1:  TopKV2(values=array([  1.00000000e+00,   3.39671054e-14,   1.61444785e-14], dtype=float32), indices=array([22,  0, 25]))
*  picture number 2:  TopKV2(values=array([  1.00000000e+00,   3.36704928e-19,   6.52081102e-25], dtype=float32), indices=array([17, 14, 20]))
*  picture number 3:  TopKV2(values=array([  1.00000000e+00,   2.93292358e-16,   7.95178158e-20], dtype=float32), indices=array([15,  2, 40]))
*  picture number 4:  TopKV2(values=array([  1.00000000e+00,   5.63426699e-14,   1.10788529e-34], dtype=float32), indices=array([11, 27, 30]))
*  picture number 5:  TopKV2(values=array([  1.00000000e+00,   2.15859703e-08,   5.24542517e-15], dtype=float32), indices=array([30, 28, 11]))
*  picture number 6:  TopKV2(values=array([  5.85210800e-01,   4.14788246e-01,   4.44743534e-07], dtype=float32), indices=array([30, 11, 29]))
*  picture number 7:  TopKV2(values=array([  1.00000000e+00,   4.99432495e-09,   3.64846278e-11], dtype=float32), indices=array([14, 39, 13]))
*  picture number 8:  TopKV2(values=array([  1.00000000e+00,   1.52759819e-11,   2.22230071e-13], dtype=float32), indices=array([14,  1, 25]))
*  picture number 9:  TopKV2(values=array([  1.00000000e+00,   4.38013394e-34,   4.22565083e-38], dtype=float32), indices=array([13, 39,  5]))
*  picture number 10:  TopKV2(values=array([  6.40612960e-01,   3.59386951e-01,   1.13161919e-11], dtype=float32), indices=array([ 6, 36,  5]))   
   As you can see from the pictures for predictions, the NN guessed the signs with 100% accuracy. 
   Looking at softmax probabilities we can see, that with 2 images probilities was not very high.
   For the 10th image (noise) probability was only 64% and for the 6 image (sign covered with snow) - 58%. Thefore to improve accuracy and reliability of the system under real-life conditions (weather, graffiti, camera malfunction etc.), the optical recognition should be combined with other methods (gps coordinates of signs, update/change signs, so that they can be recognized not only with a camera, collect data from other cars and sources). 
   
   ---
### Visualisation of the network 
   ---   
   
In this section after successfully training the neural network I want to see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
<p align="center">
<img src="readme_images\nn_inner_states.PNG" width="480" alt="nn inner states" />
</p>
   
   ---
### Possible improvements
   ---
   
Designed system recognized test/downloaded images quite good. To improve its performance the following can be done:
* using larger training sets with noisy/empty images in it to avoid situations, when NN sees traffic signs in any provided image;
* use several NN trained on different data sets with images made by different ambient conditions (winter, sommer, rain, day, night, fog etc.);
* to impelement additional ways to protect from false recognitions due to graffiti or some adverse/hostile acts. 
