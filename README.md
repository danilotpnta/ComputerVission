# ComputerVission
## Project 1 
- Apply filtering techniques(gaussian filter, bilateral filter)
- Object detection(Sobel, Hough transform) 
- Masking: merge, blur
- Morphological Operations (Dilataion, Erosion, Closing)

![alt text](https://github.com/danilotpnta/ComputerVission/blob/main/preject1.jpg)

## Project 2 
This project is divided into two task:

1. Craft and Extract face Features from four celebrities using SIFT and PCA detector
2. Train a model (kNN, SVM, RandomForest, CNN) using supervised learning on a dataset of 80 labelled images 
   1. For testing the accuracy of the model a dataset of 1136 unlabelled images where provided
   2. Kaggle was used to run a competition among several models 

##### The Dataset

| Actor Face                                                   | Label | Num Imgs |
| ------------------------------------------------------------ | ----- | -------- |
| Jesse Eisenberg                                              | 1     | 30       |
| Michael Cera (who arguably looks similar to Jesse Eisenberg) | 0     | 10       |
| Mila Kunis                                                   | 2     | 30       |
| Sarah Hyland (who arguably looks similar to Mila Kunis)      | 0     | 10       |

![alt text](https://github.com/danilotpnta/ComputerVission/blob/main/img/5.png)

##### Feature Extractors

###### SIFT

- Visual Bag of Words are used to create fingerprints of each picture
- kMeans clustering is used to narrow down the features extracted from each face
- Implement t-SNE plots to reduce dimensionality of set and fine tune hyperparamters of SIFT extractor
![alt text](https://github.com/danilotpnta/ComputerVission/blob/main/img/8.png)

###### PCA
- Reduce the dimensionality of the feature space by projecting the data onto a lower-dimension
- Use eigenfaces to reconstruct an actor's face using PCA features extractor
![alt text](https://github.com/danilotpnta/ComputerVission/blob/main/img/6.png)

##### Results
![alt text](https://github.com/danilotpnta/ComputerVission/blob/main/img/7.png)

The Ensemble method implemented use a combination of kNN, SVM, and CNN to predict the label of an image. The model shows positive results achieving the following accuracies 

| Models    | Training Set | Test Set |
| --------- | ------------ | -------- |
| Ensembles | 0.86         | 0.81     |
| kNN       | 0.83         | 0.66     |
| SVM       | 0.75         | 0.76     |
| CNN       | 0.89         | 0.78     |

Hence, it is then concluded that a combination of models using majority vote over the labels using hard constraint yields an accuracy that is optimally. Put into words, 8 out of 10 times we are able to recognise/distinguish who actor is who.


## Project 3 
### [Coming] - Deadline: 24th May
