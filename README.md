# CancelOut
# TL;DR
**CancelOut** is a layer for deep neural networks, that can help identify a subset of relevant input features for streaming or static data.  

TAGS: Feature Importance, Feature Ranking, Feature Selection, Deep Learning Sensitivity Analysis.

# Intuition 

<img src="http://vadimborisov.com/CancelOut.png" width="480">

The main idea is to update weights (W) of CancelOut during a training stage, so that ''noisy'' or less essential features will be canceled out with a negative weight. Otherwise, the best features, which contribute more to a learning process is going to be passed through with a positive weight. One can see CancelOut is a "gate" input, there NN decides who is going to pass through (see the equation below). 

![equation](https://latex.codecogs.com/gif.latex?CancelOut(\boldsymbol{X})&space;=&space;\boldsymbol{X}&space;\odot&space;g&space;({W_{CO}}))

![where](https://latex.codecogs.com/gif.latex?$\hspace{2mm}&space;where&space;$\boldsymbol{X}$&space;is&space;an&space;input&space;vector&space;$\boldsymbol{X}&space;\in&space;\mathbb{R}^N$,&space;$W_{CO}$&space;is&space;a&space;weight&space;vector&space;$W_{CO}&space;\in&space;\mathbb{R}^N$,&space;$N$&space;is&space;a&space;feature&space;size,&space;and&space;$g$&space;is&space;an&space;activation&space;function.&space;Note,&space;$g(x)$&space;denotes&space;here&space;elementwise&space;application,&space;e.g.&space;$&space;\boldsymbol{X}&space;=\begin{bmatrix}&space;a&space;\\&space;b&space;\\&space;c&space;\\&space;\end{bmatrix}&space;$,&space;then&space;$g(\boldsymbol{X})&space;=&space;g\biggl(\begin{bmatrix}&space;a&space;\\&space;b&space;\\&space;c&space;\\&space;\end{bmatrix}\biggl)&space;=&space;\biggl(\begin{bmatrix}&space;g(a)&space;\\&space;g(b)&space;\\&space;g(c)&space;\\&space;\end{bmatrix}\bigg)$.)

# Example 

For examples, please refer to the `<framework>_example.ipynb` files.  

Or just copy the code: 


## PyTorch implementation:
```python

class CancelOut(nn.Module):
    '''
    CancelOut Layer
    
    x - an input data (vector, matrix, tensor)
    '''
    def __init__(self,inp, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp,requires_grad = True) + 4)
    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))

```
## Keras/TensorFlow implementation:

```python
class CancelOut(keras.layers.Layer):
    '''
    CancelOut Layer
    '''
    def __init__(self, activation='sigmoid', cancelout_loss=True, lambda_1=0.002, lambda_2=0.001):
        super(CancelOut, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cancelout_loss = cancelout_loss
        
        if activation == 'sigmoid': self.activation = tf.sigmoid
        if activation == 'softmax': self.activation = tf.nn.softmax

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1),
            trainable=True)
        
    def call(self, inputs):
        if self.cancelout_loss:
            self.add_loss( self.lambda_1 * tf.norm(self.w, ord=1) + self.lambda_2 * tf.norm(self.w, ord=2))
        return tf.math.multiply(inputs, self.activation(self.w))
    
    def get_config(self):
        return {"activation": self.activation}    
```

#  * Work in progress. *


