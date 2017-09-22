import numpy as np
import os


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmod_derivative(x):
    return sigmod(x) * (1 - sigmod(x))


class NN:
    def __init__(self, nn, learning_rate=1e-1):
        self.learning_rate = learning_rate
        self.nn = nn
        self.__init_network()

    def fit(self, X, Y):
        self.__check_input(X, Y, self.nn)
        nws = [np.zeros(w.shape) for w in self.w]
        nbs = [np.zeros(b.shape) for b in self.b]
        # get delta
        for x, y in zip(X, Y):
            dws, dbs = self.__back_propagation(x, y)
            nws = [nw + dw for nw, dw in zip(nws, dws)]
            nbs = [nb + db for nb, db in zip(nbs, dbs)]

        # update weights and bias
        batch_size = X.shape[0]
        lr = self.learning_rate
        self.w = [w - lr * dw / batch_size for w, dw in zip(self.w, nws)]
        self.b = [b - lr * db / batch_size for b, db in zip(self.b, nbs)]

    def __back_propagation(self, x, y):
        z_list, a_list = self.__forward(x)
        dws = [np.zeros(w.shape) for w in self.w]
        dbs = [np.zeros(b.shape) for b in self.b]
        delta = (a_list[-1] - y) * sigmod_derivative(z_list[-1])
        dbs[-1] = delta
        dws[-1] = np.dot(a_list[-2].reshape(a_list[-2].shape[0], -1), delta.reshape(delta.shape[0], -1).transpose())
        layers = len(self.nn) - 1
        for layer in range(2, layers):
            delta = np.dot(self.w[-layer + 1].transpose(), delta) * sigmod_derivative(z_list[-layer])
            dws[-layer] = np.dot(a_list[-layer - 1], delta)
            dbs[-layer] = delta
        return dws, dbs

    def predict(self, X):
        for w, b in zip(self.w, self.b):
            X = np.dot(X, w) + b
            X = sigmod(X)
        return X

    def __forward(self, x):
        a = x
        z_list = []
        a_list = [a]
        for w, b in zip(self.w, self.b):
            z = np.dot(a, w) + b
            a = sigmod(z)
            z_list.append(z)
            a_list.append(a)
        return z_list, a_list

    def __init_network(self):
        self.w = []
        self.b = []
        for i in range(1, len(self.nn)):
            input = self.nn[i - 1]
            output = self.nn[i]
            self.w.append(self.__weights([input, output]))
            self.b.append(self.__bias([output]))

    def __weights(self, shape):
        return np.random.normal(scale=0.01, size=shape)

    def __bias(self, shape):
        return np.ones(shape).astype(np.float32) * 0.01

    def save(self, file_name='./model/params'):
        params = np.array([self.w, self.b])
        np.save(file_name, params)

    def restore(self, file_name='./model/params.npy'):
        if not os.path.exists(file_name):
            print 'No file exists.'
        else:
            try:
                params = np.load(file_name)
                self.w = params[0]
                self.b = params[1]
                print 'Load successfully.'
            except:
                print 'Load failure.'

    def __check_input(self, X, Y, nn):
        if nn is None or len(nn) < 2 or X.shape[-1] != nn[0] or Y.shape[-1] != nn[-1]:
            raise Exception('Input error')
