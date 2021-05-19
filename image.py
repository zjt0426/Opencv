import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(r'D:\Desktop\program\sence.jpeg')

cv2.imshow('image', image)
print(image.shape)
b, g, r = cv2.split(image)
cv2.imshow('R', r)
merge = cv2.merge([b, g, r])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow('gray', gray)
cv2.imshow('hsv', hsv)
cv2.imshow('lab', lab)

blurred = np.hstack([
    cv2.GaussianBlur(image, (7, 7), 0),
    cv2.GaussianBlur(image, (3, 3), 0),
    cv2.GaussianBlur(image, (3, 3), 0)

])

cv2.imshow('blurred', blurred)

blurred_ = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred_, 30, 100)

cv2.imshow('Canny', canny)
canny1 = cv2.Canny(blurred_, 30, 50)
cv2.imshow('Canny1', canny1)


chans = cv2.split(image)

colors = ('b', 'g', 'r')
for (chan, color) in zip(chans,colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.show()


new_width, new_height = 200, 250
resized_image = cv2.resize(image,dsize=(new_width,new_height),interpolation=cv2.INTER_AREA)

cv2.imshow('resized_image', resized_image)

flipped_image = cv2.flip(image, 1)

cv2.imshow('flipped_image', flipped_image)

mask = np.zeros(image.shape[:2], dtype="uint8")
(x, y) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (x-100, y-100), (x+100, y+100), 255, -1)
cv2.imshow('mask', mask)

masked = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('masked', masked)


patch = image[50:300, 100:400]
patch[50:100, 50:120] = (135, 156, 175)
cv2.imshow('patch', patch)




cv2.waitKey()

