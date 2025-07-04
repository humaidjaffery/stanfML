import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
from ucimlrepo import fetch_ucirepo 


class logisticRegression:
    def __init__(self, x, y, learning_rate):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.coefficient = [0, 0]
    
    def scratch(self):
        track_cost = []
        track_prediction = []
        track_coefficients_0 = []
        track_coefficients_1 = []

        for epoch in range(5000):
            cost = [0, 0]
            for i in range(len(self.x)):
                actual = self.y[i]
                predicted = 1 / (1 + math.exp(- (self.coefficient[0] + self.coefficient[1] * self.x[i] )))
                cost[0] += (actual - predicted)
                cost[1] += (actual - predicted) * self.x[i]
            #gradient ascent
            cost[0] /= len(self.x)
            cost[1] /= len(self.x)
            self.coefficient[0] += self.learning_rate * cost[0]
            self.coefficient[1] += self.learning_rate * cost[1]

            track_cost.append(cost[1])
            track_coefficients_0.append(self.coefficient[0])
            track_coefficients_1.append(self.coefficient[1])
            track_prediction.append(predicted)
            print(cost)

        plt.subplot(1, 3, 1)
        plt.plot(range(len(track_cost)), track_cost)
        plt.title("Cost")
        plt.subplot(1, 3, 2)
        plt.plot(range(len(track_coefficients_0)), track_coefficients_0)
        plt.title("Coefficient[0]")
        plt.subplot(1, 3, 3)
        plt.plot(range(len(track_coefficients_1)), track_coefficients_1)
        plt.title("Coefficient[1]")
        plt.show()
    
    def graph_model(self):
        predictions = []
        for input in self.x:
            predictions.append(1 / (1 + math.exp(- (self.coefficient[0] + self.coefficient[1] * input ))))
        
        plt.plot(self.x, predictions)
        plt.show()

    def predict(self, petal_width):
        return 1 / (1 + math.exp(- (self.coefficient[0] + self.coefficient[1] * petal_width )))


class __init__:
  
    # fetch dataset 
    iris = fetch_ucirepo(id=53) 

    # data (as pandas dataframes) 
    X = iris.data.features 
    y = iris.data.targets  

    # variable information 
    print(y.get("class"))
    print(X.get("petal width")) 

    # plt.subplot(2, 2, 1)
    # plt.title("Petal length")
    # plt.plot(X.get("petal length"), y.get("class"))

    # plt.subplot(2, 2, 2)
    # plt.title("Petal width")
    # plt.plot(X.get("petal width"), y.get("class"))

    # plt.subplot(2, 2, 3)
    # plt.title("sepal length")
    # plt.plot(X.get("sepal length"), y.get("class"))

    # plt.subplot(2, 2, 4)
    # plt.title("Sepal Width")
    # plt.plot(X.get("sepal width"), y.get("class"))
    # plt.show()

    # "iris-setosa" "Iris-versicolor" "Iris-virginica"
    final_y = []
    final_x = []
    for i in range(len(list(y.get("class")))):
        if y.loc[i].get("class") == "Iris-setosa":
            final_y.append(0)
            final_x.append(X.get("petal width").loc[i])
        elif y.loc[i].get("class") == "Iris-versicolor":
            final_y.append(1)
            final_x.append(X.get("petal width").loc[i])
        elif y.loc[i].get("class") == "Iris-virginica":
            pass
            
    print(pd.Series(final_y).shape)

    print("Starting lecture 3 (logistic regression)...")

    logreg = logisticRegression(pd.Series(final_x), pd.Series(final_y), 0.1)
    logreg.scratch()
    logreg.graph_model()
    p = logreg.predict(2.0)
    print(p)





    