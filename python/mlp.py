import time
import numpy as np
import matplotlib.pyplot as plt


def load_data(dir_name):
    """
    Function for loading MNIST data stored in comma delimited files. Labels for
    each image are the first entry in each row.

    Parameters
    ----------
    dit_name : str
         Path to where data is contained

    Returns
    -------
    X : array_like
        A (N x p=784) matrix of samples
    Y : array_like
        A (N x 1) matrix of labels for each sample
    """
    data = list()  # init a list called `data`

    with open(dir_name, "r") as f:  # open the directory as a read ("r"), call it `f`
        for line in f:  # iterate through each `line` in `f`
            split_line = np.array(
                line.split(",")
            )  # split lines by `,` - cast the resultant list into an numpy array
            split_line = split_line.astype(
                np.float32
            )  # make the numpy array of str into floats
            data.append(split_line)  # collect the sample into the `data` list

    data = np.asarray(
        data
    )  # convert the `data` list into a numpy array for easier indexing

    # as the first number in each sample is the label (0-9), extract that from the rest and return both (X,Y)
    return data[:, 1:], data[:, 0]


X_train, Y_train = load_data("data/mnist_train.csv")
X_test, Y_test = load_data("data/mnist_test.csv")

# rescale data between 0 - 1.0
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()

# one-hot encode train (y_train) and test (y_test) set labels
y_train = np.zeros((Y_train.size, int(Y_train.max()) + 1))
y_train[np.arange(Y_train.size), Y_train.astype(np.int32)] = 1.0

y_test = np.zeros((Y_test.size, int(Y_test.max()) + 1))
y_test[np.arange(Y_test.size), Y_test.astype(np.int32)] = 1.0

num_images = 4
fig, axes = plt.subplots(1, num_images, figsize=(15, 10))
for image, label, ax in zip(X_train[:num_images], y_train[:num_images], axes):
    ax.imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1.0)
    ax.set_title(f"Label: {label}")


class MLP:
    def __init__(self, X, Y, X_val, Y_val, L=1, N_l=128):
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1) # add bias
        self.Y = np.squeeze(np.eye(10)[Y.astype(np.int32).reshape(-1)])
        print(self.Y)
        self.X_val = np.concatenate((X_val, np.ones((X_val.shape[0], 1))), axis=1) # add bias
        self.Y_val = np.squeeze(np.eye(10)[Y_val.astype(np.int32).reshape(-1)])
        self.L = L # number of hidden layers
        self.N_l = N_l # number of neurons in each hidden layer
        self.n_samples = self.X.shape[0]
        self.layer_sizes = np.array([self.X.shape[1]] + [N_l] * L + [self.Y.shape[1]]) # sizes of all layers
        self.__init_weights()
        self.train_loss = list()
        self.train_acc = list()
        self.val_loss = list()
        self.val_acc = list()
        self.train_time = list()
        self.tot_time = list()
        self.metrics = [
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.train_time,
            self.tot_time,
        ]

    def __sigmoid(self, x):
        # VCompute the sigmoid
        return 1.0 / (1.0 + np.exp(-x))

    def __softmax(self, x):
        # Compute softmax along the rows of the input
        exponent = np.exp(x)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def __loss(self, y_pred, y):
        # Compute the loss along the rows, averaging along the number of samples
        return ((-np.log(y_pred)) * y).sum(axis=1).mean()

    def __accuracy(self, y_pred, y):
        # Compute the accuracy along the rows, averaging along the number of samples
        return np.all(y_pred == y, axis=1).mean()

    def __sigmoid_prime(self, h):
        # Compute the derivative of sigmoid where h=sigmoid(x)
        return h * (1 - h)

    def __to_categorical(self, x):
        # Transform probabilities into categorical predictions row-wise, by simply taking the max probability
        categorical = np.zeros((x.shape[0], self.Y.shape[1]))
        categorical[np.arange(x.shape[0]), x.argmax(axis=1)] = 1
        return categorical

    def __init_weights(self):
        # Initialize the weights of the network given the sizes of the layers
        self.weights = list()
        for i in range(self.layer_sizes.shape[0] - 1):
            self.weights.append(
                np.random.uniform(
                    -1, 1, size=[self.layer_sizes[i], self.layer_sizes[i + 1]]
                )
            )
            print(self.weights[-1].shape)
        # self.weights = np.asarray(self.weights)

    def __init_layers(self, batch_size):
        # Initialize and allocate arrays for the hidden layer activations
        self.__h = [np.empty((batch_size, layer)) for layer in self.layer_sizes]

    def __feed_forward(self, batch):
        # Perform a forward pass of `batch` samples (N_samples x N_features)
        h_l = batch
        self.__h[0] = h_l
        for i, weights in enumerate(self.weights):
            h_l = self.__sigmoid(h_l.dot(weights))
            self.__h[i + 1] = h_l
        self.__out = self.__softmax(self.__h[-1])

    def __back_prop(self, batch_y):
        # Update the weights of the network through back-propagation
        delta_t = (self.__out - batch_y) * self.__sigmoid_prime(self.__h[-1])
        for i in range(1, len(self.weights) + 1):
            self.weights[-i] -= (
                self.lr * (self.__h[-i - 1].T.dot(delta_t)) / self.batch_size
            )
            delta_t = self.__sigmoid_prime(self.__h[-i - 1]) * (
                delta_t.dot(self.weights[-i].T)
            )

    def predict(self, X):
        # Generate a categorical, one-hot, prediction given an input X
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.__init_layers(X.shape[0])
        self.__feed_forward(X)
        return self.__to_categorical(self.__out)

    def evaluate(self, X, Y):
        # Evaluate the performance (accuracy) predicting on X with true labels Y
        prediction = self.predict(X)
        return self.__accuracy(prediction, Y)

    def train(self, batch_size=8, epochs=25, lr=1.0):
        # Train the model with a given batch size, epochs, and learning rate. Store and print relevant metrics.
        self.lr = lr
        self.batch_size = batch_size
        for epoch in range(epochs):
            start = time.time()

            self.__init_layers(self.batch_size)
            shuffle = np.random.permutation(self.n_samples)
            train_loss = 0
            train_acc = 0
            X_batches = np.array_split(
                self.X[shuffle], self.n_samples / self.batch_size
            )
            Y_batches = np.array_split(
                self.Y[shuffle], self.n_samples / self.batch_size
            )
            for batch_x, batch_y in zip(X_batches, Y_batches):
                self.__feed_forward(batch_x)
                train_loss += self.__loss(self.__out, batch_y)
                train_acc += self.__accuracy(self.__to_categorical(self.__out), batch_y)
                self.__back_prop(batch_y)

            train_loss = train_loss / len(X_batches)
            train_acc = train_acc / len(X_batches)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            train_time = round(time.time() - start, 3)
            self.train_time.append(train_time)

            self.__init_layers(self.X_val.shape[0])
            self.__feed_forward(self.X_val)
            val_loss = self.__loss(self.__out, self.Y_val)
            val_acc = self.__accuracy(self.__to_categorical(self.__out), self.Y_val)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            tot_time = round(time.time() - start, 3)
            self.tot_time.append(tot_time)

            print(
                f"Epoch {epoch+1}: loss = {train_loss.round(3)} | acc = {train_acc.round(3)} | val_loss = {val_loss.round(3)} | val_acc = {val_acc.round(3)} | train_time = {train_time} | tot_time = {tot_time}"
            )


model = MLP(X_train, Y_train, X_test, Y_test, L=1, N_l=128)
model.train(batch_size=8, epochs=25, lr=1.0)
