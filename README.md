# Speech Emotion Recognition using combination of MLP & lightgbm using VotingCLassifier 

Monitoring emotional state of people and extending automated support to them has geared up the quality of human computer interaction. This is highly relevant especially in pandemic situations where people are forced to remain indoors. This work proposes a system to capture the voice of a person, detect the emotion and respond back with appropriate response that would give him emotional support.

## Accuracy:
* MLP - 68%
* combination of MLP & lightgbm - 70%

# Methods
![](images/audio.jpg?raw=true)

The proposed system is used to record a voice clip on hotword detected, preprocessed, feature extracted, classified and responded according the detected emotion. The responses can be created and altered according to the environment and use case. Once a certain number of live audio is captured, this could be used to train the model for further accuracy.

### *run command 'python app.py' in terminal and activate hotword by saying either 'computer' or 'alexa'.*

## This repository contains the following:

* Major_Project_MLP.ipynb - Python notebook to train model (with RAVDESS dataset)
* Emotion_Voice_Detection_Model.h5 - pretrained model
* app.py - hotword detection program
* recorder.py - records audio and saves it to a file
* feature.py - extracts feature from audio file
* emotion.py - predicts and returns emotion from audio file
* X.train - features of training set required for StandardScaling

![](images/test.png?raw=true)

The python notebook can be used to train an MLP model or a combination of MLP and lightgbm to achieve an accuracy of 68% and 70& respectively.

![](images/confusion_matrix_test.jpg?raw=true)
