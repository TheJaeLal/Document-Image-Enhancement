#HyperPrameters
epochs,k1,k2 = (100,1,1)

import cv2
import numpy as np


def find_gradient(I):
	return

def calculate_delta(I,J,k1,k2):

	grad_I = find_gradient(I)
	grad_J = find_gradient(J)
	
	dw = 2*k1*I*(J-255) + 2*grad_I*(grad_J - grad_I)
	db = 2*k1*(J-255)

	return dw,db

def update_weights(w,dw,b,db,alpha):
	w = w - alpha*dw
	b = b - alpha*db
	return w,b

I = cv2.imread('Images/src.jpg')
I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

height,weight = I.shape

b = np.zeros((height,weight))
w = np.ones((height,weight))

#Initialize output image with zeros (completely black)
J = np.zeros((height,weight))

for e in range(epochs):
	J = w*I + b
	dw,db = calculate_delta(I,J,k1,k2)
	w,b = update_weights(w,dw,b,db,alpha=0.5)

cv2.imwrite('Output.jpg',J)