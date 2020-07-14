import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    Ix2 = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=5)
    Iy2 = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=5)
    It = im2 - im1

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    list1 = []
    list2 = []

    for i in range(win_size, im2.shape[0] - win_size, step_size):
        for j in range(win_size, im2.shape[1] - win_size, step_size):
            Ix = Ix2[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            Iy = Iy2[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            Itu = It[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten() * -1
            Ai = np.vstack((Ix, Iy)).T
            b = np.reshape(Itu, (Itu.shape[0], 1))
            d = np.dot(np.linalg.pinv(Ai), b)

            u[i, j] = d[0]
            v[i, j] = d[1]

            #if d[0] >= d[1] and d[0] > 0 and (d[0] / d[1]) < 100:

            list1.append((d[0] , d[1]))
            list2.append((j, i))
    return np.asarray(list2), np.asarray(list1)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaus_kernel = cv2.getGaussianKernel(5, -1)
    gaus_kernel = np.multiply(gaus_kernel, gaus_kernel.transpose()) * 4.
    gaus_pyr = gaussianPyr(img, levels)
    lap_pyr = []
    for i in range(levels-1):
        gau_expand = gaussExpand(gaus_pyr[i + 1], gaus_kernel)
        gau_expand = gau_expand[:gaus_pyr[i].shape[0], :gaus_pyr[i].shape[1]]
        lap = gaus_pyr[i] - gau_expand
        lap_pyr.append(lap)
    lap_pyr.append(gaus_pyr[i+1])
    return lap_pyr

def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaus_kernel = cv2.getGaussianKernel(5, -1)
    gaus_kernel = np.multiply(gaus_kernel, gaus_kernel.transpose()) * 4.
    restored = lap_pyr[-1]
    lap_pyr = np.flip(lap_pyr)
    for i in range(1, len(lap_pyr)):
        expand = gaussExpand(restored, gaus_kernel)
        expand = expand[:lap_pyr[i].shape[0], :lap_pyr[i].shape[1]]
        restored = expand + lap_pyr[i]

    return restored

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    img_copy = img.copy()
    gaussian_kernel = cv2.getGaussianKernel(5, -1)
    gaussian_kernel = np.multiply(gaussian_kernel, gaussian_kernel.transpose())
    gaus = [img_copy]
    for i in range(1, levels):
        img_copy = cv2.filter2D(img_copy, -1, gaussian_kernel)
        img_copy = img_copy[::2, ::2]
        gaus.append(img_copy)
    return gaus



def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    G = np.zeros(img.shape, dtype=np.float64)
    G = cv2.resize(G, (img.shape[1] * 2, img.shape[0] * 2))
    G[::2, ::2] = img
    return cv2.filter2D(G, -1, gs_k)

def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    lap_img_1 = laplaceianReduce(img_1, levels)
    lap_img_2 = laplaceianReduce(img_2, levels)
    gau_mask = gaussianPyr(mask, levels)
    naive_blend = img_1*mask + (1 - mask) * img_2
    lap_bend = []
    for i in range(levels):
        lap_bend.append(gau_mask[i] * lap_img_1[i] + (1 - gau_mask[i]) * lap_img_2[i])
    out = laplaceianExpand(lap_bend)
    return naive_blend, out
