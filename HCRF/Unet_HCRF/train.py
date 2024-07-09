from keras.models import *
from keras.layers import *
from keras.optimizers import *
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io

###################################################
##Function to create folder
def makedir(path):

    folder = os.path.exists(path)#Check if folder exsists

    if folder is True:
        print("Folder already Exists")
    else:
        os.makedirs(path)#Create folder
        print("Creating Folder")
        print("Done")
###################################################


###################################################
             ####   Very Important ####

file_path = "E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/Unet_augmentation"
i = m_class  # Microorganism category, from 0 to 20 classes
n = train_num  # The n-th training


image_size = (256, 256, 1)
data_size = (256, 256)


test_num = 280  # Number of test images
BATCH_SIZE = 8

Aug_originall = file_path + "/" + str(i) + "/aug/original_" + str(n)  # Augmentation folder for training dataset
# Define the path to create the folder, where the original images are stored
Aug_GTM1 = file_path + "/" + str(i) + "/aug/GTM_" + str(n)
# Define the path to create the folder, where the ground truth of original images is stored, each time different datasets are augmented
Aug_original2 = file_path + "/" + str(i) + "/aug/val_original_" + str(n) # Augmentation folder for validation dataset
Aug_GTM2 = file_path + "/" + str(i) + "/aug/val_GTM_" + str(n)
makedir(Aug_originall)
makedir(Aug_GTM1)
makedir(Aug_original2)
makedir(Aug_GTM2)
train_path = file_path + "/" + str(i) + "/train"
test_path = file_path + "/" + str(i) + "/test"
val_path = file_path + "/" + str(i) + "/val"
##################################################

