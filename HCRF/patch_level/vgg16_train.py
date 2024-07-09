import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
# num_classes=2
# def mycrossentropy(y_true, y_pred, e=0.14):
#        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

"""
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
    # Be sure to update the storage location!!
    com.to_csv('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\history.csv')


# Plot accuracy and loss curve
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # Change the figure title here!!
    plt.title("vgg16 model")
    plt.ylabel("acc-loss")
    plt.xlabel("epoch")
    plt.legend([" acc", "val acc", " loss", "val loss"], loc="upper right")
    # plt.show()
    #Be sure to update the storage name!!!
    plt.savefig("E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\vgg16_model.png")


if __name__ == "__main__":
    start = time.perf_counter()
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)## Round down, distribute training images into the network
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    #Data augmentation, see https: // blog.csdn.net / jacke121 / article / details / 79245732
    #val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE)
    #Using the flow_from_directory method to read image data from the hard drive
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                              batch_size=BATCH_SIZE)
    ## Used for batch processing of data
    classes = list(iter(batches.class_indices))

    # Build the pre-trained model without a classifier (include_top=False) and only extract features
    base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


    for layer in base_model.layers:
        layer.trainable = False   # Lock all InceptionV3 convolutional layers
    Inp = Input((64, 64, 3))
    x = base_model(Inp)
    # x = base_model(Inp).layers[-1].output
    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(x)
    # # Add a fully connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Add a classifier, assuming we have classes length classes
    predictions = Dense(len(classes), activation="softmax")(x)
    # Build the complete model we need to train
    finetuned_model = Model(inputs=Inp, outputs=predictions)
    # First, we only train the top few layers (randomly initialized layers)
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=15)

    checkpointer = ModelCheckpoint('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\vgg16_best.h5',
                                   verbose=1, save_best_only=True)
    # Train a few epochs on the new dataset, keeping the base_model parameters unchanged, only the last layer parameters are updated

    History = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=15,
                                            callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                                            validation_steps=num_valid_steps)
    # end1 = time.clock()
    # print("save model before", end1)
    # finetuned_model.save('D:\\PycharmProject\\GastricCarcinoma\\vgg16_best.h5')
    save_history(History)
    plot_history(History)

    end2 = time.perf_counter()
    print("final is in ", end2 - start) 