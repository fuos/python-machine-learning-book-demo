import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    #eta:学习速率(0~1)
    #n_iter:迭代次数(10)
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    #
    def fit(self, x, y):
        #
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0

            #
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    # predict function    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',
                header=None)

y=df.iloc[0:100, 4].values
y=np.where(y == 'Iris-versicolor', -1 ,1)
x=df.iloc[0:100,[0,2]].values

#数据集散点图
plt.scatter(x[:50,0],x[:50,1],
          color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],
          color='yellow',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')

#训练感知器
ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(x,y)

#迭代收敛次数图
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.show()
