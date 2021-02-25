# Do not import any additional 3rd party external libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import softmax

class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example (do not change)

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, linear):
        # sigmoid
        self.x = x
        self.linear = linear
        print('sig input x shape', self.x.shape)
        self.saved = 1/(1+np.exp(-self.linear))
        return self.saved

    def derivative(self):
        # deriv of sigmoid
        return self.saved * (1-self.saved)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # tanh(x)
        self.saved = (2/(1+np.exp(-2*x))) - 1
        return self.saved 

    def derivative(self):
        # 1 - (tanh(x))^2
        return 1-(self.saved**2)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.saved = max(0,x)
        return self.saved

    def derivative(self):
        return self.saved * (self.saved >= 0)


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.loss = None
        self.y, self.y_hat = None, None
        # you can add variables if needed

    def forward(self, x, y):
        print("Doing CE loss")
        N = y.shape[0]

        self.y_hat = softmax(x, axis=1) # after softmax 
        self.y = y

        print("softmaxed y_hat", self.y_hat.shape)
        print("actual y\n", self.y.shape)

        y_idx = self.y.argmax(axis=1)

        # print("selected", self.y_hat[range(N), y_idx]) # selects y_hats corresp to actual y
        log_likelihood = -np.log(self.y_hat[range(N), y_idx])
        self.loss = np.sum(log_likelihood) / N
        print(f'loss over {N} samples: {self.loss}')
        return self.loss

    def derivative(self):
        # softmax(linear out) - actual
        print("ce deriv", self.y_hat.shape, self.y.shape)
        return self.y_hat - self.y


# randomly intialize the weight matrix with dimension d0 x d1 via Normal distribution
def random_normal_weight_init(d0, d1):
    W = np.random.normal(size=(d0,d1))
    return W


# initialize a d-dimensional bias vector with all zeros
def zeros_bias_init(d):
    b = np.zeros((1,d))
    return b


class MLP(object):

    """
    A simple multilayer perceptron
    (feel free to add class functions if needed)
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr):

        self.num_hiddens = len(hiddens)
        # pre/post activation, hidden/output
        self.pre_h = [0]*self.num_hiddens
        self.post_h = [0]*self.num_hiddens
        self.pre_o, self.post_o = 0, 0
        self.softmax = None

        # Don't change this -->
        self.train_mode = True
        self.nlayers = self.num_hiddens + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes
        self.nn_dim = [input_size] + hiddens + [output_size]
        # list containing Weight matrices of each layer, each should be a np.array
        self.W = [weight_init_fn(self.nn_dim[i], self.nn_dim[i+1]) for i in range(self.nlayers)]

        for i in range(len(self.W)):
            print(f"W_{i} is {self.W[i].shape}")
        
        # list containing derivative of Weight matrices of each layer, each should be a np.array
        self.dW = [np.zeros_like(weight) for weight in self.W]
        # list containing bias vector of each layer, each should be a np.array
        self.b = [bias_init_fn(self.nn_dim[i+1]) for i in range(self.nlayers)]
        # list containing derivative of bias vector of each layer, each should be a np.array
        self.db = [np.zeros_like(bias) for bias in self.b]

        # You can add more variables if needed

    def forward(self, x):
        # hidden node: 784 -> 128
        # print("input")
        for i in range(self.num_hiddens):
            # linear
            # print('x: ', x.shape)
            # print('W:', self.W[i].shape)
            self.pre_h[i] = np.dot(x, self.W[i])
            # activation function
            self.post_h[i] = self.activations[i].forward(x, self.pre_h[i])
            # print("Post", self.post_h[i].shape)
        # output layer: 128 -> 10
        # -1 should be nlayers - 1
        self.pre_o = np.dot(self.post_h[-1], self.W[-1])
        # print("before softmax:", self.pre_o)
        # self.post_o = self.activations[-1].forward(self.pre_o)

        # self.post_o = softmax(self.pre_o, axis=1)
        print('linear before softmax', self.pre_o.shape)

    def zero_grads(self):
        # set dW and db to be zero
        self.dW = [np.zeros_like(weight) for weight in self.W]
        self.db = [np.zeros_like(bias) for bias in self.b]

    def step(self):     
        # update the W and b on each layer
        print(self.W[0].shape, self.dW[0].shape)
        for i in range(self.nlayers):
            self.W[i] = self.W[i] - self.lr * self.dW[i]
            self.b[i] = self.b[i] - self.lr * self.db[i]

    def backward(self, labels):
        self.loss = self.criterion.forward(self.pre_o, labels)
        if self.train_mode:
            # calculate dW and db only under training mode
            sftmax_deriv = self.criterion.derivative()
            
            # W1, b1
            self.dW[1] = sftmax_deriv.T @ self.post_h[0]
            print(self.dW[1].shape)
            self.db[1] = sftmax_deriv

            #W0, b0
            x = self.activations[0].x
            print('layer 0')
            print(sftmax_deriv.shape)
            print(self.W[1].shape)
            print(self.activations[0].derivative().shape)
            print(x.shape)
            self.dW[0] = sftmax_deriv @ self.W[1].T @ self.activations[0].derivative().T @ x
            self.db[0] = sftmax_deriv @ self.W[1].T @ self.activations[0].derivative().T 
             
            

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        # training mode
        self.train_mode = True

    def eval(self):
        # evaluation mode
        self.train_mode = False

    def get_loss(self, labels):
        # return the current loss value given labels
        return self.loss

    def get_error(self, labels):
        # return the number of incorrect preidctions gievn labels
        return NotImplementedError

    def save_model(self, path='p1_model.npz'):
        # save the parameters of MLP (do not change)
        np.savez(path, self.W, self.b)


# Don't change this function
def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    for e in range(nepochs):
        print("epoch: ", e)
        train_loss = 0
        train_error = 0
        val_loss = 0
        val_error = 0
        num_train = len(trainx)
        num_val = len(valx)

        for b in range(0, num_train, batch_size):
            mlp.train()
            mlp.forward(trainx[b:b+batch_size])
            mlp.backward(trainy[b:b+batch_size])
            mlp.step()
            train_loss += mlp.get_loss(trainy[b:b+batch_size])
            train_error += mlp.get_error(trainy[b:b+batch_size])
        training_losses += [train_loss/num_train]
        training_errors += [train_error/num_train]
        print("training loss: ", train_loss/num_train)
        print("training error: ", train_error/num_train)

        for b in range(0, num_val, batch_size):
            mlp.eval()
            mlp(valx[b:b+batch_size])
            val_loss += mlp.get_loss(valy[b:b+batch_size])
            val_error += mlp.get_error(valy[b:b+batch_size])
        validation_losses += [val_loss/num_val]
        validation_errors += [val_error/num_val]
        print("validation loss: ", val_loss/num_val)
        print("validation error: ", val_error/num_val)

    test_loss = 0
    test_error = 0
    num_test = len(testx)
    for b in range(0, num_test, batch_size):
        mlp.eval()
        mlp(testx[b:b+batch_size])
        test_loss += mlp.get_loss(testy[b:b+batch_size])
        test_error += mlp.get_error(testy[b:b+batch_size])
    test_loss /= num_test
    test_error /= num_test
    print("test loss: ", test_loss)
    print("test error: ", test_error)

    return (training_losses, training_errors, validation_losses, validation_errors)


# get ont hot key encoding of the label (no need to change this function)
def get_one_hot(in_array, one_hot_dim):
    dim = in_array.shape[0]
    out_array = np.zeros((dim, one_hot_dim))
    for i in range(dim):
        idx = int(in_array[i])
        out_array[i, idx] = 1
    return out_array


def main():
    # load the mnist dataset from csv files
    image_size = 28 # width and length of mnist image
    num_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    # train_data = np.loadtxt("../mnist/mnist_train.csv", delimiter=",")
    # test_data = np.loadtxt("../mnist/mnist_test.csv", delimiter=",") 
    # np.savez_compressed("../mnist/compressed.npz",train=train_data,test=test_data)

    npz_file = np.load("../mnist/compressed.npz")
    train_data = npz_file['train']
    test_data = npz_file['test']

    # rescale image from 0-255 to 0-1
    fac = 1.0 / 255
    train_imgs = np.asfarray(train_data[:50000, 1:]) * fac
    val_imgs = np.asfarray(train_data[50000:, 1:]) * fac
    test_imgs = np.asfarray(test_data[:, 1:]) * fac
    train_labels = np.asfarray(train_data[:50000, :1])
    val_labels = np.asfarray(train_data[50000:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # convert labels to one-hot-key encoding
    train_labels = get_one_hot(train_labels, num_labels)
    val_labels = get_one_hot(val_labels, num_labels)
    test_labels = get_one_hot(test_labels, num_labels)

    print(train_imgs.shape)
    print(train_labels.shape)
    print(val_imgs.shape)
    print(val_labels.shape)
    print(test_imgs.shape)
    print(test_labels.shape)

    dataset = [
        [train_imgs, train_labels],
        [val_imgs, val_labels],
        [test_imgs, test_labels]
    ]

    # These are only examples of parameters you can start with
    # you can tune these parameters to improve the performance of your MLP
    # this is the only part you need to change in main() function
    hiddens = [128]
    activations = [Sigmoid()]
    lr = 0.05
    num_epochs = 10
    batch_size = 8

    # build your MLP model
    mlp = MLP(
        input_size=image_pixels, 
        output_size=num_labels, 
        hiddens=hiddens, 
        activations=activations, 
        weight_init_fn=random_normal_weight_init, 
        bias_init_fn=zeros_bias_init, 
        criterion=SoftmaxCrossEntropy(), 
        lr=lr
    )

    # train the neural network
    losses = get_training_stats(mlp, dataset, num_epochs, batch_size)

    # save the parameters
    mlp.save_model()

    # visualize the training and validation loss with epochs
    training_losses, training_errors, validation_losses, validation_errors = losses

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(training_losses, color='blue', label="training")
    ax1.plot(validation_losses, color='red', label='validation')
    ax1.set_title('Loss during training')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(training_errors, color='blue', label="training")
    ax2.plot(validation_errors, color='red', label="validation")
    ax2.set_title('Error during training')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('error')
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()