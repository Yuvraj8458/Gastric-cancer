# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io
import glob
import scipy.io as sio


#Normalize images, prepare ground truth, separate foreground and background
def adjustData(original,mask):
    original = original/255
    mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask < 0.5] =0
    return(original,mask)


# Generate training images
def trainGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,#Convert images to grayscale
            target_size = target_size,#data_size = (256, 256) resize images
            batch_size = batch_size, #BATCH_SIZE = 8
            save_to_dir = aug_image_save_dir,#Save augmented images to directory
            save_prefix = original_aug_prefix,
            seed = seed)
    # myGene = trainGenerator(BATCH_SIZE, train_path, type, "GTM", aug_dict, target_size=data_size,
                                #aug_image_save_dir=Aug_originall, aug_mask_save_dir=Aug_GTM1)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)

    train_generator = zip(original_generator,mask_generator)

    ## Pair original images with their corresponding masks
    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)
#Generation of validation images
def validationGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_image_save_dir,
            save_prefix = original_aug_prefix,
            seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)



    train_generator = zip(original_generator,mask_generator)



    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)


def testGenerator(test_path, num_image, target_size):
    for pngfile in glob.glob(test_path + "/*.png"):#pngfile
        img = cv.imread(pngfile, cv.IMREAD_GRAYSCALE)#OpenCV"IMREAD_UNCHANGED"、"IMREAD_GRAYSCALE"、"IMREAD_COLOR"
        img = img / 255
        img = cv.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img

def saveResult(save_path, result, flag_multi_class=False, num_class=2):
    for i, item in enumerate(result):
        img = item[:, :, 0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        # cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)  # opencvimwrite
        # img = cv.resize(img, (2048,2048))
        img = np.array(img)

        # cv.imwrite(save_path + "/" + str(i) + "_predict.mat", img)
        # sio.savemat(save_path + "/" + ("%04d" % i) + ".mat", {'img': img}) #val
        sio.savemat(save_path + "/" + ("%05d" % i) + ".mat", {'img': img})  # val

