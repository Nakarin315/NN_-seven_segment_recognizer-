"""
Created on Wed Dec  8 22:00:35 2021
Reference: https://youtu.be/w8yWXqWQYmU (https://www.kaggle.com/c/digit-recognizer)
@author: Nakarin Jayjong

"""
import os
import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# get the training data set from CSV file
data = pd.read_csv('C:/Users/nakar/Desktop/NN_digit_reconizer/database.csv')
data = np.array(data)
m, n = data.shape
n_Node = 30; # Number of nodes
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[700:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

def init_params():
    W1 = np.random.rand(n_Node, 784) - 0.5
    b1 = np.random.rand(n_Node, 1) - 0.5
    W2 = np.random.rand(n_Node, n_Node) - 0.5
    b2 = np.random.rand(n_Node, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    # one_hot_Y = np.zeros((Y.size, Y.max() + 1));
    one_hot_Y = np.zeros((Y.size,n_Node));
    one_hot_Y[np.arange(Y.size), Y] = 1;
    one_hot_Y = one_hot_Y.T;
    return one_hot_Y;

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y);
    dZ2 = A2 - one_hot_Y;
    dW2 = 1 / m * dZ2.dot(A1.T);
    db2 = 1 / m * np.sum(dZ2);
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1);
    dW1 = 1 / m * dZ1.dot(X.T);
    db1 = 1 / m * np.sum(dZ1);
    return dW1, db1, dW2, db2;

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;   
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;    
    return W1, b1, W2, b2;

def get_predictions(A2):
    return np.argmax(A2, 0);

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size;

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X);
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y);
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
        # if i % 10 == 0: # You can see the accuracy of the prediction
        #     predictions = get_predictions(A2);
        #     print("Iteration: ", i)
        #     print('Accuracy:',round(get_accuracy(predictions, Y)*100,1),'%')
    return W1, b1, W2, b2

# Training the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions



left, top, right, bottom =[2]*4 # border widths;
direct = 'E:/Singapore/git_hub/git_hub_digit_recognizer/database/final'
wsize = int(28) # x-axis pixel
hsize = int(28) # y-axis pixel
i_fig=1;
# Get the real image which took from the lab and test the model 
for filename in os.listdir(direct):
    if filename.endswith(".jpg"):
        plt.figure(i_fig); i_fig+=1;
        print('File name: ',filename)
        img = cv2.imread(os.path.join(direct,filename))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(np.uint8(img),(wsize,wsize))
        plt.imshow(img)
        img = np.array(img[:,:,2]) 
        img = img .reshape((784, 1))/255
        prediction = make_predictions(img, W1, b1, W2, b2)
        print("Prediction: ", prediction)
