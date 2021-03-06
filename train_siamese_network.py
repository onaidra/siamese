# import the necessary packages
from tensorflow.python.keras.layers.core import Dropout
from siamese_network import build_siamese_model
from utils import Generator
import config
import utils
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
pred = Dropout(0.2)(outputs)
#model = Model(inputs=[imgA, imgB], outputs=pred)ù
model = ResNet50(include_top=False, weights='imagenet', input_tensor=tf.concat([imgA, imgB],1), pooling=max)

print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",	metrics=["accuracy"])
# train the model
"""
train_datagen = Generator(trainX, trainY, config.BATCH_SIZE)
test_datagen = Generator(testX, testY, config.BATCH_SIZE)

print("[INFO] training model...")

history = model.fit_generator(train_datagen,
    steps_per_epoch=len(trainX)//config.BATCH_SIZE,
    validation_data=test_datagen,
    validation_steps=len(testX)//config.BATCH_SIZE,
    epochs=config.EPOCHS)
"""
print(pairTest[:,0].shape)
print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE,epochs=config.EPOCHS)


# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)
