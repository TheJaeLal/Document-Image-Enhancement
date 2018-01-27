import cv2
import numpy as np

src = cv2.imread('Images/src.jpg')

src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

height,weight = src.shape

b = np.zeros((height,weight))
w = np.ones((height,weight))

dest = w*src + b

cv2.imwrite('Output.jpg',dest)