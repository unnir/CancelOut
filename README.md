# CancelOut

**CancelOut** is a layer for deep neural networks, that can help identify a subset of relevant input features for streaming or static data.  The CancelOut layers allows you to make a Feature Importance analysis. 

TAGS: Feature Importance, Feature Selection, Deep Learning Sensitivity Analysis.

# Intuition 

<img src="http://vadimborisov.com/CancelOut.png" width="480">

The main idea is to update weights (W) of CancelOut during a training stage, so that ''noisy'' or less essential features will be canceled out with a negative weight. Otherwise, the best features, which contribute more to a learning process is going to be passed through with a positive weight. One can see CancelOut is a "gate" input, there NN decides who is going to pass through (see the equation below). 

![equation](https://latex.codecogs.com/gif.latex?CancelOut%28%5Cboldsymbol%7BX%7D%29%20%3D%20%5Cboldsymbol%7BX%7D%20%5Codot%20%5Csigma%28%5Cmathcal%7BW%7D%29%20%3D%20%5Cboldsymbol%7BX%7D%20%5Codot%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-%5Cmathcal%7BW%7D%7D%7D)


# Example 

For examples, please refer to the `example.ipynb` file.  


#  * Work in progress. *

### TODO:
- [x] PyTorch implementation
- [ ] Keras / TensorFlow implementation 
- [ ] More examples 
