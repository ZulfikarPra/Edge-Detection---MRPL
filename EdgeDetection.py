import cv2 as cv
import numpy as np

Image = cv.imread('imageToProcess.jpg')
grayscaleImage = cv.cvtColor(Image, cv.COLOR_BGR2GRAY).astype(np.float64)
cv.imwrite("grayscaleImage.png", grayscaleImage)

BlackWhiteImage = cv.threshold(grayscaleImage, 55, 255, cv.THRESH_BINARY)[1].astype(np.float64)

cv.imwrite("BlackWhiteImage.png", BlackWhiteImage)

x = -1*np.array([[-1,0,1]])
Dx = cv.filter2D(BlackWhiteImage, -1, x)
cv.imwrite("Dx.png",np.abs(Dx))

y = -1*np.array([[-1],[0],[1]])
Dy = cv.filter2D(BlackWhiteImage, -1, y)
cv.imwrite("Dy.png",np.abs(Dy))

magnitudeL2 = np.sqrt(Dx**2 + Dy**2)

cv.imwrite("edgeDetection.png",magnitudeL2)