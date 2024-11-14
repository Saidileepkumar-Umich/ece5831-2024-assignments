
import numpy as np
from layers import Relu, Affine, SoftmaxWithLoss

class TwoLayerNetWithBackProp:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = {}
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.loss_layer = SoftmaxWithLoss()
        self.grads = {}
    
    def predict(self, x):
        x = self.layers['Affine1'].forward(x)
        x = self.layers['Relu1'].forward(x)
        x = self.layers['Affine2'].forward(x)
        return x
    
    def forward(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.layers['Affine2'].backward(dout)
        dout = self.layers['Relu1'].backward(dout)
        dout = self.layers['Affine1'].backward(dout)
        
        self.grads['W1'] = self.layers['Affine1'].dW
        self.grads['b1'] = self.layers['Affine1'].db
        self.grads['W2'] = self.layers['Affine2'].dW
        self.grads['b2'] = self.layers['Affine2'].db
        return dout
