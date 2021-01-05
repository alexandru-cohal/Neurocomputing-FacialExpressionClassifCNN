# Neurocomputing-FacialExpressionClassifCNN
### *"A Convolutional Neural Network Solution For Facial Expression Classification"* project developed for the *"Neurocomputing"* course within the *EMECS* Masters

- **Date**: March 2018
- **Purpose**: The purpose of this project is to develop a solution based on a *Convolutional Neural Network* for classifying the neutral state and 5 facial expressions (left wink / blink, right wink / blink, strong blink, open mouth, full mouth) of a person using the 14 channels data provided by the *Emotiv EPOC Headset*
- **Programming Language**: Python & Tensorflow framework
- **Team**: Individual project
- **Inputs**:
  - The 14 channels data provided by the *Emotiv EPOC Headset* (monitoring the electrical activity of the muscle tissue)
- **Outputs**:
  - The facial state (neutral, left wink / blink, right wink / blink, strong blink, open mouth, full mouth)
- **Solution**:
  - Time-series CNN is used
  - The input of the CNN is a 2D array because time-windows are used (data from individual time instances can be affected by noise)
  - The CNN used has:
    - 1 input layer
    - 4 1D convolution layers
    - 4 1D pooling layer
    - 1 fully connected layer
  - The performance of the CNN is evaluated through the values of:
    - Loss
    - Accuracy
    - Training time
- **Results**:
  - The best result obtained had a testing accuracy of *95%*, a training time of *41.86 seconds* and a loss of approximately *0.1*
  - For more information about the solution, implementation, results, conclusions and improvements see [this document](documentation/Neurocomputing-FacialExpressionClassifCNN-Documentation.pdf)
