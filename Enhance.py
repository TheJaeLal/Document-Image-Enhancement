#HyperPrameters
epochs,alpha,k1,k2 = (20,0.01,1,2)

import cv2
import numpy as np


def find_gradient(I):
	kernel_x = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
	kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	grad_x = cv2.filter2D(I,-1,kernel_x)
	grad_y = cv2.filter2D(I,-1,kernel_y)

	return np.array([grad_x,grad_y])

def calculate_delta(I,J,k1,k2):

	grad_I = find_gradient(I)
	grad_J = find_gradient(J)
	
	result = grad_I*(grad_J - grad_I)
	result_x = result[0]
	result_y = result[1]

	new_result = result_x+result_y
	
	print("Shape of new_result = {}".format(new_result.shape))


	dw = 2*k1*I*(J-255) + k2*2*new_result
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
	print("Epoch {}".format(e))
	dw,db = calculate_delta(I,J,k1,k2)
	w,b = update_weights(w,dw,b,db,alpha)
	J = w*I + b
	cv2.imwrite("Outputs/"+str(e)+".jpg",J)

#cv2.imwrite('Output.jpg',J)