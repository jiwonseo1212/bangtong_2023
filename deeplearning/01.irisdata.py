import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def prepare_data(target):
    iris = load_iris()
    X_tr = iris.data[:, 2:]
    labels  = iris.target_names  # setosa : 0, versicolor :1 , virginica: 2
    y = iris.target

    y_tr =  np.array(
                    [labels[y[i]] == target for i in range(len(iris.target))], 
                    dtype=int)
    return X_tr, y_tr, ["(1)" + target, "(0) the others"]

def step(x):
    return int(x >= 0)


class Perceptron:
    def __init__(self, dim, activation):
        rnd = np.random.default_rng() #난수 생성기
        self.dim = dim
        self.activation = activation

        self.w = rnd.normal(scale=np.sqrt(2.0/dim), size=dim)
        self.b = rnd.normal(scale=np.sqrt(2.0/dim))


    def printW(self):
        #가중치와 바이어스 값 출력"
        for i in range(self.dim):
            print(' w{} = {:6.3f}'.format(i+1, self.w[i]), end="")
        print(' b = {:6.3f}'.format(self.b))

    def predict(self, x):
        return np.array(
                    [
            self.activation(
                np.dot(self.w, x[i])+ self.b)
                    for i in range(len(x))
            ]
        )

    

    def fit(self, X, y, N,  epochs, eta=0.01):
        #학습표본의 인덱스를 무작위 순서로 섞음
        idx = list(range(N))
        np.random.shuffle(idx)
        X = np.array([X[idx[i]] for i in range(N)])
        y = np.array([y[idx[i]] for i in range(N)])

        f = "Epochs = {:4d}\n Loss = {:8.5f}"
        print("w의 초깃값", end="")
        self.printW()
        for j in range(epochs):
            for i in range(N):
                delta = self.predict([X[i]])[0] - y[i]
                self.w -= eta * delta * X[i] 
                self.b -= eta * delta


            if j < 10 or (j+1) % 100 == 0:
                loss = self.predict(X) - y
                loss = (loss * loss).sum() / N
                print(f.format(j+1, loss), end="")
                self.printW()        

if __name__ == "__main__":
    nSample = 150
    nDim = 2
    target = "virginica"
    X_tr, y_tr, labels = prepare_data(target)

    perceptron = Perceptron(nDim, activation=step)
    perceptron.fit(X_tr, y_tr, nSample, epochs=1000, eta=0.01)


