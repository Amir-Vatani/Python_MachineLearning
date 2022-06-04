import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, learning_rate=0.05, epochs=2):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X_train, Y_train):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        
        N = self.X_train.shape[0]

        self.w = np.random.rand(2, 1)
        fig = plt.figure(figsize=(12, 6))
        
        self.Errors = []

        x_range = np.arange(self.X_train[:,0].min(), self.X_train[:,0].max())
        y_range = np.arange(self.X_train[:,1].min(), self.X_train[:,1].max())
        
        for self.epochs in range(self.epochs):
            for i in range(N):
                y_pred = np.matmul(self.X_train[i,:], self.w)
                e = self.Y_train[i] - y_pred
  
                x = self.X_train[i,:].reshape(-1, 1)
                self.w += self.learning_rate * e * x

                fig.clear()
                Y_pred = np.matmul(self.X_train, self.w)
                ax1 = fig.add_subplot(121,projection='3d')
                ax1.clear()
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], self.Y_train, c='#0000ff')

                x, y = np.meshgrid(x_range, y_range)
                z = self.w[0] * x + self.w[1] * y
                ax1.plot_surface(x, y, z, alpha=0.4)
                ax1.set_xlabel("CRIM")
                ax1.set_ylabel("TAX")
                ax1.set_zlabel("MEDV")

                Error = np.mean(np.abs(self.Y_train - Y_pred))
                self.Errors.append(Error)

                ax2 = fig.add_subplot(122)
                ax2.clear()
                ax2.plot(np.arange(0,i+1), self.Errors)
                ax2.set_title('Train')
                
                plt.pause(0.01)
        plt.show()

    def predict(self, X_test):
        Y_pred = np.matmul(X_test, self.w)
        return Y_pred
    
    def evaluate(self, X, Y):
        Y_pred = np.matmul(X, self.w)
        Error = np.abs(Y - Y_pred)
        MSE = np.mean(Error**2)
        return MSE

    
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['CRIM', 'TAX']]
Y = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, Y)

percep = Perceptron(0.00001, 1)
percep.fit(X, Y)

MSE = percep.evaluate(X_test, y_test)

print('MSE = ' ,MSE)