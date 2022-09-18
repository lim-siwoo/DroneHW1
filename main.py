#
# Drone and Robotics Homework #1
#
# I am _____________________
#

import numpy as np
import cv2

def convertRGBtoGray(rgb):
    gray  = rgb[:,:,0] * 0.114 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.299
    gray = np.squeeze(gray).astype(np.uint8)
    return gray

def brightness(gray, val):
    res_img = gray.copy() * 0

    # your code here
    res_img = np.where((255 - gray) < 100,255,gray+val)

    return res_img

def contrast(gray, grad, inter):
    res_img = gray.copy() * 0

    # your code here
    res_img = np.clip((1+grad)*gray - 128*grad, 0, 255).astype(np.uint8)
    brightness(res_img, inter)
    return res_img

def scaling1(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype)

    # your code here (uncomment below)
    #  forward warping
    # for r in range(h):
    #     for c in range(w):

    return res_img

def scaling2(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype)

    # your code here (uncomment below)
    #  backward warping
    # for r in range(h*s):
    #     for c in range(w*s):

    return res_img

# deg: angle
def rotation(gray, angle):
    res_img = gray.copy() * 0

    #  your code here

    return res_img



if __name__ == '__main__':

    # open image
    img_rgb = cv2.imread('image.png', cv2.IMREAD_COLOR)

    # get dimension
    h, w, ch = img_rgb.shape

    # if you want to know the dimension of img_rgb, remove comment below
    # print(img_rgb.shape)

    # mission 1 : convert color image to grayscale
    img_gray = convertRGBtoGray(img_rgb)

    # mission 2: decrease brightness
    # caution: clip values between 0 ~ 255
    img_bright = brightness(img_gray, 50)

    # mission 3: decrease brightness
    # a: gradient, b:an intercept of y axis
    img_contrast = contrast(img_gray, 1.5, -50)

    # mission 4: scaling
    # move source pixels to target
    img_scaling1 = scaling1(img_gray, 3)

    # mission 5: scaling2
    # move source pixels to target
    img_scaling2 = scaling2(img_gray, 3)

    # mission 6: rotation
    # caution: Rotate the image around the center of the image.
    img_rotation = rotation(img_gray, 30)

    #concatenate results
    img_res1 = cv2.hconcat([img_gray, img_bright, img_contrast])
    img_res2 = cv2.hconcat([img_scaling1, img_scaling2])


    # display input image & results
    cv2.imshow('input image', img_rgb)
    cv2.imshow('gray, bright, contrast', img_res1)
    cv2.imshow('scaling', img_res2)
    cv2.imshow('rotation', img_rotation)
    cv2.waitKey(0)