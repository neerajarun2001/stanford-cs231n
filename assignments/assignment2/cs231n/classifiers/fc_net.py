from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = 0
        self.params['b2'] = 0

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        h1, cache_h1 = affine_relu_forward(X, W1, b1)
        scores, cache_scores = affine_forward(h1, W2, b2)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)

        dh1, grads['W2'], grads['b2'] = affine_backward(dout, cache_scores)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dh1, cache_h1)

        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # efficiency 
        hidden_dims.append(num_classes)

        for x in range(self.num_layers):

          # weights and biases intialization
          # -- set keys
          # -- init weights as random dist centered at zero, std = weight_scale
          # -- init biases as zero
          w_key = 'W{0}'.format(x+1)
          b_key = 'b{0}'.format(x+1)

          w_val = weight_scale * np.random.randn(input_dim, hidden_dims[x])
          b_val = np.zeros(hidden_dims[x])

          self.params[w_key] = w_val
          self.params[b_key] = b_val

          input_dim = hidden_dims[x]

          # gamma and beta initialization (batch/layernorm)
          if self.normalization is not None:

            # we do not apply batchnorm to last layer
            if x == self.num_layers - 1: 
              continue

            # keys
            gamma_key = 'gamma{0}'.format(x+1)
            beta_key = 'beta{0}'.format(x+1)

            # values -- ones for gamma, zeros for beta, both shape (D, )
            gamma_val = np.ones(hidden_dims[x])
            beta_val = np.zeros(hidden_dims[x])

            self.params[gamma_key] = gamma_val
            self.params[beta_key] = beta_val
          
        hidden_dims.remove(num_classes) 



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # key into caches with 'ax', 'bx', 'rx', 'dx'
        # ax -- cache for affine layer
        # bx -- cache for batchnorm layer
        # rx -- cache for relu layer
        # dx -- cache for dropout layer
        caches = {}

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1)
        for layer in range(self.num_layers - 1):

          # retrieve the weights and biases for current layer
          w = self.params['W{0}'.format(layer+1)]
          b = self.params['b{0}'.format(layer+1)]

          # affine layer
          X, caches['a' + str(layer+1)] = affine_forward(X, w, b)

          # optional batchnorm/layernorm layer
          if self.normalization is not None: 
            
            # retrieve gamma and beta 
            gamma = self.params['gamma{0}'.format(layer+1)]
            beta = self.params['beta{0}'.format(layer+1)]

            if self.normalization == 'batchnorm':
              X, caches['b' + str(layer+1)] = batchnorm_forward(X, gamma, beta, self.bn_params[layer]) 
            elif self.normalization == 'layernorm':
              X, caches['b' + str(layer+1)] = layernorm_forward(X, gamma, beta, self.bn_params[layer]) 

          # relu activation 
          X, caches['r' + str(layer+1)] = relu_forward(X)

          # optional dropout layer 
          if self.use_dropout: 
            X, caches['d' + str(layer+1)] = dropout_forward(X, self.dropout_param)
          
        # final affine layer
        w = self.params['W{0}'.format(self.num_layers)]
        b = self.params['b{0}'.format(self.num_layers)]

        scores, caches['a' + str(self.num_layers)] = affine_forward(X, w, b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # softmax loss 
        loss, dout = softmax_loss(scores, y)

        # backprop through affine layer
        # -- set keys to index into grads
        # -- call affine_backward to get dx, dw, db
        # -- update gradients in grads for w and b

        w = 'W{0}'.format(self.num_layers)
        b = 'b{0}'.format(self.num_layers)

        dout, dw, db = affine_backward(dout, caches['a' + str(self.num_layers)])

        grads[w] = dw
        grads[b] = db

        # L2 regularization
        loss += 0.5 * self.reg * np.sum(np.square(self.params[w]))
        grads[w] += self.reg * self.params[w]     

        # backprop through {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) layers
        for layer in reversed(range(self.num_layers - 1)):

          # optional dropout layer backprop
          if self.use_dropout:
            cache = caches['d' + str(layer+1)]
            dout = dropout_backward(dout, cache)

          # backprop through relu
          cache = caches['r' + str(layer+1)]
          dout = relu_backward(dout, cache)

          # optional backprop through batchnorm (fast implementation)
          if self.normalization is not None:
            cache = caches['b' + str(layer+1)]
            if self.normalization == 'batchnorm':
              dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
            elif self.normalization == 'layernorm':
              dout, dgamma, dbeta = layernorm_backward(dout, cache)

            # keys for beta and gamma grads
            beta = 'beta{0}'.format(layer+1)
            gamma = 'gamma{0}'.format(layer+1)

            # update beta and gamma 
            grads[beta] = dbeta
            grads[gamma] = dgamma

          # backprop through affine 
          cache = caches['a' + str(layer+1)]
          dout, dw, db = affine_backward(dout, cache)
          
          # keys for params in grads dictionary
          w = 'W{0}'.format(layer+1)
          b = 'b{0}'.format(layer+1)

          # store gradients for weights and biases
          grads[w] = dw 
          grads[b] = db 

          # L2 regularization
          loss += 0.5 * self.reg * np.sum(np.square(self.params[w]))
          grads[w] += self.reg * self.params[w]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
