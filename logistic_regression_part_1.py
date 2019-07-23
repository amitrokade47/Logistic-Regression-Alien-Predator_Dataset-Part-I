# Using logistic regression with neural networks

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from PIL import ImageTk
import pprint
from sklearn.utils import shuffle


directories = ["__PASTE_YOUR_DIRECTORY_HERE__/alien_predator_data/train/predator/",
               "__PASTE_YOUR_DIRECTORY_HERE__/alien_predator_data/train/alien/",
               "__PASTE_YOUR_DIRECTORY_HERE__/alien_predator_data/validation/alien/",
               "__PASTE_YOUR_DIRECTORY_HERE__/alien_predator_data/validation/predator/"]
counter = 0

index_val = 0

#Since I know there are 347,347,100,100 images which total to 894 , I have created a matrix of 894 and will divide it
# later once I fetch all the data


list_data = np.zeros((894, 196608))


for i in range(len(directories)):
    while (True):
        #counter is used to build image_directory and find image like YOUR_DIRECTORY/1.jpg
        #Once the image is not found i.e it exxceeds the total images , it will break and go to the next directory
        # using i variable
        try:
            image_pil_format = (Image.open(directories[i] + str(counter) + ".jpg")).resize((256, 256))
            image_array = np.array(image_pil_format)
            if image_array.shape != (256, 256, 3):
                '''If image is not of RGB , i.e if image is grayscale , then it contains only 1 channel , so to 
                make it to a RGB format , add 2 blank channels of zeros behind the current channel'''
                zero = np.zeros((256, 256))
                image_array = np.dstack([image_array, zero, zero])

            image_array = (image_array / 255.0).reshape(1, -1)
            image_again = Image.fromarray(image_array.astype('uint8'))
            list_data[index_val] = image_array
            counter += 1
            index_val += 1
        except IndexError:
            counter = 0
            break
        except FileNotFoundError:
            counter = 0
            break

"""Predator:1     Alien:0"""

"""Building the X Train Data"""
predator_train = list_data[:347][:].T
alien_train = list_data[347:694][:].T
x_train = np.concatenate((predator_train, alien_train), axis=1)
# print("x train ",x_train.shape)

"""Building the Y Train Data"""
y_train_1 = np.ones((1, predator_train.shape[1]))
y_train_2 = np.zeros((1, alien_train.shape[1]))
y_train = np.concatenate((y_train_1, y_train_2), axis=1)
# print("y_train",y_train.shape)

"""Shuffle the TRAIN data so it gets mixed up and the model does not keep training initially on the same type of data"""
x_train, y_train = shuffle(x_train.T, y_train.T, random_state=24)
x_train = x_train.T
y_train = y_train.T

print("After shuffling TRAIN")
print("X train", x_train.shape)
print("Y train", y_train.shape)

"""X Test Data"""
alien_test = list_data[694:794][:].T
predator_test = list_data[794:][:].T
x_test = np.concatenate((alien_test, predator_test), axis=1)

"""Building the Y Test Data"""
y_test_1 = np.zeros((1, alien_test.shape[1]))
y_test_2 = np.ones((1, predator_test.shape[1]))

y_test = np.concatenate((y_test_1, y_test_2), axis=1)

"""Shuffle the TEST data so it gets mixed up and the model does not keep training initially on the same type of data"""
x_test, y_test = shuffle(x_test.T, y_test.T, random_state=24)
x_test = x_test.T
y_test = y_test.T

print("After shuffling TEST")
print("X test", x_test.shape)
print("Y test", y_test.shape)

abc = (np.asarray(list_3d_alien_train_images))


def create_empty_matrix(dim):
    weights = np.zeros((dim, 1))
    bias = 0
    return weights, bias

def sigmoid(matrix):
    """returns 1/1+e^-x for every val in matrix"""
    value=1/(1+np.exp(-matrix))
    return value


def forward_and_back_propogation(w, x, y, b):
    '''returns db,dz and cost'''

    """A is the prediction matrix"""

    '''A dictionary to hold values for dw,dz and cost'''
    gradient = {}
    m = x.shape[1]

    '''Forward Propogation'''
    A = sigmoid(np.dot(w.transpose(), x) + b)

    epsilon = 1e-5
    '''Original Cost Function: cost = (-1/m)*np.sum((Y*np.log(A))+(1-Y)*np.log(1-A))'''
    '''Adding a small value for epsilon to avoid underflow of values for log(A) i.e to avoid log(0)'''
    
    cost = (-1 / m) * np.sum((y * np.log(A + epsilon)) + (1 - y) * np.log(1 - A + epsilon))
    


    '''Backpropogation'''
    dz = A - y
    dw = (1 / m) * (np.dot(x, dz.T))
    db = (1 / m) * np.sum(dz)

    gradient["db"] = db
    gradient["dw"] = dw
    cost = np.squeeze(cost)
    gradient["cost"] = cost

    return gradient


def gradient_descent_optimization(w, x, y, b, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        gradient = forward_and_back_propogation(w, x, y, b)
        db = gradient["db"]
        dw = gradient["dw"]
        cost = gradient["cost"]

        '''Taking a step using gradient descent'''

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after ", i, " iterations is :", cost)

    parameters = {"w": w, "b": b}
    gradient = {"db": db, "dw": dw, "cost": cost}

    return parameters, gradient


def make_prediction(w, b, x):
    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)

    A = sigmoid(np.dot(w.T, x) + b)

    for i in range(A.shape[1]):
        y_prediction[0, i] = 0 if A[0, i] < 0.5 else 1

    return y_prediction


def train(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5):
    print(num_iterations, learning_rate)
    w, b = create_empty_matrix(x_train.shape[0])

    parameters, gradient = gradient_descent_optimization(w, x_train, y_train, b, num_iterations, learning_rate)

    cost = gradient["cost"]

    w = parameters["w"]
    b = parameters["b"]

    y_prediciton_for_train = make_prediction(w, b, x_train)
    y_prediciton_for_test = make_prediction(w, b, x_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediciton_for_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediciton_for_test - y_test)) * 100))

    d = {"costs": cost,
         "Y_prediction_test": y_prediciton_for_test,
         "Y_prediction_train": y_prediciton_for_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

d=train(x_train,y_train,x_test,y_test,num_iterations=1200,learning_rate=0.8)
