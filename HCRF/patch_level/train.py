import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import pandas as pd


""""
#1. Data paths
#2. Model save path and name, make sure to change when loading the model
#3. Modify storage path for history
#4. Store figure titles and paths
"""
DATA_DIR = 'E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')

VALID_DIR = os.path.join(DATA_DIR, 'val')


SIZE = (64, 64)
BATCH_SIZE = 35


def save_history(History):
    acc = pd.Series(History.history['accuracy'], name='accuracy')
    loss = pd.Series(History.history['loss'], name='loss')
    val_acc = pd.Series(History.history['val_accuracy'], name='val_accuracy')
    val_loss = pd.Series(History.history['val_loss'], name='val_loss')
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    # Be sure to update the storage location!
    com.to_csv('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\history.csv')
    
#plot accuracy and loss curve
def plot_history(history):
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #Change figure title
    plt.title("vgg16 model")
    plt.ylabel("acc-loss")
    plt.xlabel("epoch")
    plt.legend(["train acc","val acc","train loss","val loss"],loc="upper right")
    #plt.show()
    # Be sure to update the storage name!!
    plt.savefig("E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\loss\\vgg16 model.png")

if __name__ == "__main__":
    start = time.perf_counter()
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    
#    model = keras.applications.resnet50.ResNet50()
#    model = keras.applications.inception_v3.InceptionV3()
    model = keras.applications.vgg16.VGG16()
#    model = keras.applications.xception.Xception()

    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    
    checkpointer = ModelCheckpoint('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\vgg16-best.h5', verbose=1, save_best_only=True)

    History=finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=10, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    #end1 = time.clock()
    #print("save model before", end1)
    finetuned_model.save('D:\\PycharmProject\\GastricCarcinoma\\vgg16\\vgg16-best.h5')
    save_history(History)
    plot_history(History)

    end2 = time.perf_counter()
    print("final is in ", end2 - start)