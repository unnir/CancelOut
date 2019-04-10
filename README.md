# CancelOut

**CancelOut** is a layer for deep neural networks, that can help identify a subset of relevant input features for streaming or static data.  The CancelOut layers allows you to make a Feature Importance analysis. 

TAGS: Feature Importance, Feature Ranking, Feature Selection, Deep Learning Sensitivity Analysis.

# Intuition 

<img src="http://vadimborisov.com/CancelOut.png" width="480">

The main idea is to update weights (W) of CancelOut during a training stage, so that ''noisy'' or less essential features will be canceled out with a negative weight. Otherwise, the best features, which contribute more to a learning process is going to be passed through with a positive weight. One can see CancelOut is a "gate" input, there NN decides who is going to pass through (see the equation below). 

![equation](https://latex.codecogs.com/gif.latex?CancelOut(\boldsymbol{X})&space;=&space;\boldsymbol{X}&space;\odot&space;g&space;({W_{CO}}))

![where](https://latex.codecogs.com/gif.latex?$\hspace{2mm}&space;where&space;$\boldsymbol{X}$&space;is&space;an&space;input&space;vector&space;$\boldsymbol{X}&space;\in&space;\mathbb{R}^N$,&space;$W_{CO}$&space;is&space;a&space;weight&space;vector&space;$W_{CO}&space;\in&space;\mathbb{R}^N$,&space;$N$&space;is&space;a&space;feature&space;size,&space;and&space;$g$&space;is&space;an&space;activation&space;function.&space;Note,&space;$g(x)$&space;denotes&space;here&space;elementwise&space;application,&space;e.g.&space;$&space;\boldsymbol{X}&space;=\begin{bmatrix}&space;a&space;\\&space;b&space;\\&space;c&space;\\&space;\end{bmatrix}&space;$,&space;then&space;$g(\boldsymbol{X})&space;=&space;g\biggl(\begin{bmatrix}&space;a&space;\\&space;b&space;\\&space;c&space;\\&space;\end{bmatrix}\biggl)&space;=&space;\biggl(\begin{bmatrix}&space;g(a)&space;\\&space;g(b)&space;\\&space;g(c)&space;\\&space;\end{bmatrix}\bigg)$.)

# Example 

For examples, please refer to the `example.ipynb` file.  


#  * Work in progress. *

### TODO:
- [x] PyTorch implementation
- [ ] Add more activation functions -> softmax, tanh 
- [ ] Keras / TensorFlow implementation 
- [ ] More examples 
