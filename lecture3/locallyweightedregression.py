import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math    

from ucimlrepo import fetch_ucirepo 


class LocallyWeightedRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tau = 5
        self.learning_rate = 0.0001
        print("init")
        print(self.x.shape)
        print(self.y.shape)
    
    def scikit_implementation(self, input):
        weights = []
        model = LinearRegression().fit(self.x, self.y, weights)
    
    def scratch(self, input):
        track_theta = []
        track_cost = []
        track_predictions = []
        
        theta = 0
        for epoch in range(50):
            cost_derived = 0
            for i in range(len(self.x)):
                weight =  math.exp( - (((self.x[i] - input) ** 2) / (2 * (self.tau ** 2)))  )
                
                if epoch == 0:
                    print(f"{weight} -- {self.x[i]} vs {input}")
                
                prediction = theta * self.x[i]
                actual = self.y[i]
                cost_derived += weight * ((prediction - actual) * self.x[i])

                # loss += weight * ((self.y[i] - (theta * self.x[i])) ** 2)
                # print("Cost Added: " + str(weight * ((self.y[i] - (theta * self.x[i])) ** 2)))
  
            theta -= (self.learning_rate * cost_derived)
            
            
            print("------- End of iteration " + str(epoch) + "------")
            print(cost_derived)
            track_cost.append(cost_derived)
            track_theta.append(theta)
            track_predictions.append(prediction)

            
        print(theta * input     )
        print(self.y[np.where(self.x == 26)[0]])


        plt.subplot(2, 2, 1)
        plt.plot(range(len(track_cost)), track_cost)
        plt.xlabel("cost")
        plt.ylabel("epoch")
        plt.title("Cost")

        plt.subplot(2, 2, 2)
        plt.plot(range(len(track_theta)), track_theta)
        # plt.title("Theta")
        plt.xlabel("theta")
        plt.ylabel("epoch")

        plt.subplot(2, 2, 3)
        plt.plot(range(len(track_predictions)), track_predictions)
        plt.title(f"Prediction (real: {self.y[256]})")
        plt.xlabel("prediction")
        plt.ylabel("epoch")

        plt.subplot(2, 2, 4)
        plt.plot(self.x, self.y, 'o')
        model_output = []
        for a in self.x:
            model_output.append(a * theta)
        plt.plot(self.x, model_output)
        plt.title("Model")
        plt.xlabel("Absences")
        plt.ylabel("Grade")

        # plt.show()


class __init__:
    print("Starting lecture 3 (locally weighed regression )...")
    # fetch dataset 
    student_performance = fetch_ucirepo(id=320) 
    
    # data (as pandas dataframes) 
    X = student_performance.data.features 
    y = student_performance.data.targets 

    print(X)

    plt.plot(X.absences, y.G3, 'o')
    plt.xlabel("Final Grade")
    plt.ylabel("Absences")
    plt.show()

    lcw = LocallyWeightedRegression(X.absences, y.G3)
    lcw.scratch(26)



        
    