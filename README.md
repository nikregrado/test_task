# test_task

Solution for Data Science Bowl 2018 (https://www.kaggle.com/c/data-science-bowl-2018/overview)
The main objective of this project is to build a Convolutional Neural Network on Unet Architeture. This is my first expirience with CNN and Unet, please don't judge to hard.

Model was build by Keras, for backend used Tensorflow(cpu version).

___

# Requirements

All requirements that you need are in requirments.txt

Also you python 3.5 interpreter
And you need jupyter notebook or if you use pycharm just import notebook from this repo to your project

___

# Overview

1.EDA for TEST.ipynb

  In EDA I'm using a k-means for clusstering train images by dominant colors, and display masks with min,max,median objects

___

2.data_reading.py

  script to load data and pre-processing for this data
  
___

3. model_unet.py 
script with model and metrics

___

4.predict_and train.py

  script that run model training and prediction, than encode each nuclei predicted mask to rle and write to submission.csv id of picture and rle encoded mask
  
___

# P.S.

This is my first expirience with CNN and Unet architecture. Thnak you for apportunity, this task hard but very intersting. Hope you understand my code and code, copyed from resourse that you give me.

