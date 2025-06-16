import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math    

#Linear Regression (Scikit Learn)
#Scikit learn uses the normal equation (A.K.A OLS: Ordinary Least squares)
class lin_reg_sk:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print()

    def run(self):
        self.model = LinearRegression()
        self.model.fit(self.x, self.y)
        predictions = self.model.predict(self.x)
        mse = mean_squared_error(predictions, self.y)
        score = self.model.score(self.x, self.y)
        print("MSE: " + str(mse))
        print("Score: " + str(score))
        print("Params: " + str(self.model.get_params()))

    def predict_score(self, x, y):
        predictions = self.model.predict(x)
        print("R_squared: " + str(r2_score(y, predictions)))


# Batch Gradient Descent (Scratch) - Only two parameters
class batch_gradient_descent:
    def __init__(self, x, y):
        self.learning_rate = 0.00001
        self.x = x
        self.y = y 
        self.parameters = [0, 0]

    def run(self):
        m = len(self.x)
        cost = []
        r2 = []

        for n in range(500):
            print("-----" + str(n) + "-----")
            print(self.parameters)
            mse = 0
            cost_derived = [0, 0]
            predictions = []

            for i in range(len(self.x)):
                prediction = self.parameters[0] + (self.parameters[1] * self.x[i])
                predictions.append(prediction)
                actual = self.y[i]
                mse += (prediction - actual) ** 2
                cost_derived[0] += (prediction - actual)
                cost_derived[1] += ((prediction - actual) * self.x[i])

            cost_derived[0] = (cost_derived[0] / m)
            cost_derived[1] = (cost_derived[1] / m)
            mse = (mse / (2*m))
            cost.append(mse)
            r2.append(r2_score(self.y, predictions))
            self.parameters[0] -= (self.learning_rate * cost_derived[0])
            self.parameters[1] -= (self.learning_rate * cost_derived[1])

            print(self.learning_rate)
            print(self.learning_rate * cost_derived[0])
           
            print("cost_derived[0]: " + str(cost_derived[0]) + "\n cost_derived[1]" + str(cost_derived[1]))
            print("Parameter[0]: " + str(self.parameters[0]))
            print("Parameter[1]: " + str(self.parameters[1]))
            print("Change[0]: " + str(self.learning_rate * cost_derived[0]))
            print("Change[1]: " + str(self.learning_rate * cost_derived[1]))
            print("MSE: " + str(mse))
            print("R2 score: " + str(r2_score(self.y, predictions)))

            if n % 50 == 0 and n > 50:
                plt.plot(self.x, self.y, 'o')
                plt.plot(self.x, predictions)
                plt.show()

        print(cost)
        print("R Sqaured: ")
        print(r2)

        plt.plot(list(range(len(cost))), cost)
        plt.show()


        
#Stochastic Gradient Descent by scratch
class stochastic_gradient_descent:
    def __init__(self, x, y):
        self.learning_rate = 0.001
        self.x = x
        self.y = y 
        self.parameters = [0, 0]

    def run(self):
        m = len(self.x)
        r2 = []
        parameter_zero = []
        parameter_one = []

        for i in range(16):
            print(m)
            print("-----" + str(i) + "-----")
            prediction = self.parameters[0] + (self.parameters[1] * self.x[i])
            actual = self.y[i]
            cost_derived = [(prediction - actual), ((prediction - actual) * self.x[i])]

            self.parameters[0] -= (self.learning_rate * cost_derived[0])
            self.parameters[1] -= (self.learning_rate * cost_derived[1])
                       
            print("cost_derived[0]: " + str(cost_derived[0]) + "\n cost_derived[1]" + str(cost_derived[1]))
            print("Parameter[0]: " + str(self.parameters[0]))
            print("Parameter[1]: " + str(self.parameters[1]))
            print("Change[0]: " + str(self.learning_rate * cost_derived[0]))
            print("Change[1]: " + str(self.learning_rate * cost_derived[1]))
            

            y = self.parameters[1] + self.parameters[1] * self.x 

            r2.append(r2_score(self.y, y))
            parameter_zero.append(self.parameters[0])
            parameter_one.append(self.parameters[1])

            # plt.plot(self.x, self.y, 'o')
            # plt.plot(self.x, y)
            # plt.show()
        plt.subplot(1, 3, 1)
        plt.plot(parameter_zero)
        
        plt.subplot(1, 3, 2)
        plt.plot(parameter_one)

        plt.subplot(1, 3, 3)
        plt.title("R squared Score")
        plt.plot(r2)

        plt.show()

class __init__:
    print("Starting...")
    ins_data = pd.read_csv('./insurance.csv')
    age = pd.Series(ins_data['age']).values.reshape(-1, 1)
    bmi = pd.Series(ins_data['bmi']).values.reshape(-1, 1)
    charges = pd.Series(ins_data['charges']).values.reshape(-1, 1)
    children = pd.Series(ins_data['children']).values.reshape(-1, 1)

    simple_data = pd.read_csv("./LR.csv")
    simple_data_test = pd.read_csv("./test.csv")
    simple_data_test.dropna(inplace=True)
    simple_data.dropna(inplace=True)
    
    x = pd.Series(simple_data['X']).values.reshape(-1, 1)
    y = pd.Series(simple_data['Y']).values.reshape(-1, 1)

    # plt.plot(x, y, 'o')
    # plt.plot(bmi, charges, 'o')
    # plt.show()

    l = lin_reg_sk(x, y)
    # l.run() 
    # l.predict_score(pd.Series(simple_data_test['x']).values.reshape(-1, 1), pd.Series(simple_data_test['y']).values.reshape(-1, 1))
    b = batch_gradient_descent(age, charges)
    # b.run()
    s = stochastic_gradient_descent(x, y)
    s.run()
