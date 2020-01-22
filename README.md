# Video_Classification_Keras
This project is about the implementation of a hybrid CNN-LSTM architecture to classify different classes of videos. 

This repository is created to recognize actions in videos by implementing Long Short Term Memory - LSTM neworks and Convolutional Neural Network (CNN). LSTM is used to capture the data in temporal form and CNN is used to capture the spatial features.

## Dependencies
* Python 3.6 (or greater)
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [Keras](https://keras.io/)
* [h5py](http://docs.h5py.org/en/stable/)

## Methodology
Every video is first converted into image frames. The features of the image frames are extracted from the last fully connected layer of VGG16() architecture. The Features are then fed into the LSTM network in a sequential manner to predict the final target action in the video. 

## Dataset
This repository is trained on a sequence of images and the dataset is available at http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html.

## Execution
1. First extract the filenames of the dataset into a csv file by running extract_csv.py
2. Extract the image frames from videos by running extract_frames.py
3. Run model_train.py

## Folder Structure
After extracting frames folder structure is as follows:

data_frames

  bend
  
    daria_bend
    
      bend_daria_bend_1.jpg
      bend_daria_bend_2.jpg
        .....
        
    denis_bend
    
      bend_denis_bend_1.jpg
      bend_denis_bend_1.jpg
      .......
      
  run
  
    daria_run
    
      run_daria_run_1.jpg
      run_daria_run_2.jpg
      .........
  ...........      
