# -*- coding: utf-8 -*-
# @Time    : ${20200627} ${12:13}
# @Author  : Changhao Sun
# @FileName: $ main.py
# @Software: $ HCRF
from model import*
from data import*
import os
import matplotlib.pyplot as plt
import math
import time

def training_vis(history,save_fig_path):
    loss = history.history['loss']
    print("loss:",loss)
    acc = history.history['acc']
    print("accuracy:",acc)
    val_loss = history.history['val_loss']
    print("validation_loss:",val_loss)
    val_acc = history.history['val_acc']
    print("validation_accuracy:",val_acc)
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label =  "train_loss")
    ax1.plot(val_loss,label = "validation_loss")
    ax1.set_title("train model loss")
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")

    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label =  "train_accuracy")
    ax2.plot(val_acc,label = "validation_accuracy")
    ax2.set_title("train model accuracy")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    plt.legend(["train","validation"], loc="upper right")


    plt.savefig(save_fig_path)

    #plt.show()
    #plt.close(fig)
    with open(save_fig_path+'.txt','a', encoding='utf-8') as f:

        f.write("loss:"+str(loss)+"\n"+"accuracy:"+str(acc)+"\n"+"validation_loss:"+str(val_loss)+"\n"+"validation_accuracy:"+str(val_acc))

def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("Folder already exists")
    else:
        os.makedirs(path)
        print("Create folder")
        print("Done")

aug_dict = dict(rotation_range=0.02,  
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')
image_size = (256, 256, 1)
data_size = (256, 256)
file_path = "E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/Unet_augmentation"
train = False 
predict = True 

def main(m_class, train_num):
    i = m_class  
    n = train_num  
    type = "original"
    test_num = 8960  
    BATCH_SIZE = 8




    #epochs = 10  
    #steps_per_epoch = 300  
    if train:
        start = time.perf_counter()
        # Aug_originall = file_path + "/" + str(i) + "/aug/original" + str(n)
        # Aug_GTM1 = file_path + "/" + str(i) + "/aug/GTM_" + str(n)
        # Aug_original2 = file_path + "/" + str(i) + "/aug/val_original_" + str(n)
        # Aug_GTM2 = file_path + "/" + str(i) + "/aug/val_GTM_" + str(n)
        # makedir(Aug_originall)
        # makedir(Aug_GTM1)
        # makedir(Aug_original2)
        # makedir(Aug_GTM2)
        Aug_originall = None
        Aug_GTM1 = None
        Aug_original2 = None
        Aug_GTM2 = None

        train_path = file_path + "/" + str(i) + "/train"
        test_path = file_path + "/" + str(i) + "/test"
        val_path = file_path + "/" + str(i) + "/val"






        myGene = trainGenerator(BATCH_SIZE, train_path, type, "GTM", aug_dict, target_size=data_size,
                                aug_image_save_dir=Aug_originall, aug_mask_save_dir=Aug_GTM1)
        valGene = validationGenerator(BATCH_SIZE, val_path, type, "GTM", aug_dict, target_size=data_size,
                                aug_image_save_dir=Aug_original2, aug_mask_save_dir=Aug_GTM2)



        # num_train_samples = sum([len(files) for r, d, files in os.walk(Aug_originall)]) 
        # num_valid_samples = sum([len(files) for r, d, files in os.walk(Aug_original2)]) 
        # print(num_train_samples,num_valid_samples)
        #
        # num_train_steps = math.floor(num_train_samples / BATCH_SIZE)  
        # num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)
        # print(num_train_steps,num_valid_steps)



        model = unet()
        model_path = test_path + "/model/"+type
        makedir(model_path)
        model_checkpoint = ModelCheckpoint(model_path + "/unet_membrane" + str(n) + ".hdf5", monitor='val_loss',mode = 'min', verbose=1,
                                           save_best_only=True)
        history = model.fit_generator(myGene, steps_per_epoch = 5000,epochs=40, validation_data=valGene,validation_steps=5000,callbacks=[model_checkpoint])
        training_vis(history,test_path + "/model/"+type+"/loss&acc" + str(n) + ".png")
        end = time.perf_counter()
        print("train is in ", end - start)

    test_path = file_path + "/" + str(i) + "/test"
    model_path = test_path + "/model/" + type
    original_val_path=r"E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/Unet/1/val"
    if predict:
        start2 = time.perf_counter()
        model = load_model(model_path + "/unet_membrane" + str(n) + ".hdf5")
        testGene = testGenerator(original_val_path + "/" + "unet_val_patch", num_image=test_num, target_size=data_size)
        #testGene = testGenerator(test_path + "/"+type, num_image=test_num, target_size=data_size)
        results = model.predict_generator(testGene, test_num, verbose=1)
        predict_save_path = original_val_path + "/predict/"+ type+"/" + str(n)
        makedir(predict_save_path)
        saveResult(predict_save_path, results)
        end2 = time.perf_counter()
        print("test is in ", end2 - start2)