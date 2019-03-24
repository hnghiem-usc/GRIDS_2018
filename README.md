# GRIDS_2018
Project for the Graduate Rising in Data Science 2018. Cancer screening.
# INTRODUCTION 
This is the first time I participated in GRIDS as a member. My project is fairly straightfowad: it is 
a classification task to predict whether a patient will develop cancer based on the demographic predictors. 
# DATA
The original data is hosted at the UCIRVINE Machine Learning Repository: 
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# NETWORK ARCHITECTURE
Besides the input layer, there is only one hidden laye FC of 5 neurons to 1 output neuron, using the 
sigmoid activation function. BATCH GRADIENT DESCENT is used to optimize the loss function (binary cross entropy).
TensorFlow is the framwwork of choice for implmentation. 
