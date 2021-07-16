import mxnet as mx
import numpy as np
import pickle
import cv2
import random

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dicts = pickle.load(f, encoding='bytes')
    images = dicts[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dicts[b'labels']
    imagearray = mx.nd.array(images)
    #labelarray = mx.nd.array(labels)
    return imagearray, labels

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    return dict[b'label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)



imgarray, lblarray = extractImagesAndLabels("cifar-10-batches-py/", "data_batch_1")
categories = extractCategories("cifar-10-batches-py/", "batches.meta")

lookup = {b'airplane': 0,
          b'automobile': 1,
          b'bird': 2,
          b'cat': 3,
          b'deer': 4,
          b'dog': 5,
          b'frog': 6,
          b'horse': 7,
          b'ship': 8,
          b'truck': 9}

rands = []

def get_rand():
    r = random.randint(0, 100001)
    while r in rands:
        r = random.randint(0, 100001)
    rands.append(r)
    return r

for i, x in enumerate(imgarray):
    cat_num = lookup[categories[int(lblarray[i])]]
    name = f'{cat_num}_{get_rand()}'
    saveCifarImage(imgarray[i], "./out/", name)
