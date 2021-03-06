import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
import tensorflow as tf
import argparse
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
tf.python.control_flow_ops = tf
import os
from scipy import ndimage, misc
import cv2
import Image
import pandas as pd
import matplotlib.pyplot as plt



N_CATEGORIAS = 100


def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size)
    
    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
print("[INFO] describing images...", args)
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the data matrix and labels list
data1 = []
labels = []
ingredientes = []
name_ingredients=[]
train = pd.read_csv('data/ingredients1.csv')
#print train.columns
name_img_ingredients=np.array(train.ix[0:,10])
#print name_img_ingredients
image_test = []
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
        # print label
 	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	feature = image_to_feature_vector(image)
	data1.append(feature)
	index= (np.where(name_img_ingredients==imagePath.split(os.path.sep)[-1]))[0][0]
	#print name_img_ingredients[index]," " ,imagePath.split(os.path.sep)[-1]
	ingredient =np.array(train.ix[index,0:10])	
	ingredient[ingredient != 1.0]=0
        print ingredient
	#print "tamano ing",len(ingredient)		      
	# show an update every 1,000 images
	ingredientes.append(ingredient)
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
#print name_ingredients
#labels = train.ix[0:,11].values.astype('int32')
#print "labels '/n " , labels
#print "shape", image_to_feature_vector(image).shape


ingredientes=np.array(ingredientes)

N_INGRADIENTS = len(ingredient)
print N_INGRADIENTS
N_CATEGORIAS = 50

images = np.asarray(data1)
N = len(images)
#print "data1", data1
#images = np.random.normal(size=[N, 32, 32, 3])
#print "images1", images
#print ingredientes
#ingredientes = np.random.uniform(low=0, high=1, size=[N, N_INGRADIENTS])
#print ingredientes.shape
#print ingredientes
#categorias = np.random.uniform(low=0, high=1, size=[N, N_CATEGORIAS])

input_image = Input(shape=[32, 32, 3])

	# create model

x = Convolution2D(20, 3, 3)(input_image)
x = Activation('relu')(x)
x = Convolution2D(20, 3, 3)(x)
x = Activation('relu')(x)
x = Convolution2D(20, 3, 3)(x)
x = Activation('relu')(x)
x = Convolution2D(20, 3, 3)(x)
x = Activation('relu')(x)
x = Convolution2D(20, 3, 3)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
out_conv = Flatten()(x)

x = Dense(32)(out_conv)
x = Activation('relu')(x)
x = Dropout(0.4)(x)

out_ingredientes = Dense(N_INGRADIENTS, activation='sigmoid', name='ingredientes')(x)




model = Model(input=input_image, output=[out_ingredientes])

	
model.compile(loss={'ingredientes': 'binary_crossentropy'},
              optimizer='adam',
              metrics={'ingredientes': 'accuracy'})

history=model.fit(x=images, y={'ingredientes': ingredientes}, validation_split=0.20,nb_epoch=300, batch_size=25)
print (history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/weight.h5")
print("Saved model to disk")



# classify the image
print("[INFO] classifying image...")

for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
 	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	feature = image_to_feature_vector(image)
	feature= np.expand_dims(feature, axis=0)
	preds = model.predict(np.array(feature))
	#print " ",label
	ingred_r=preds[0]
        #print ingred_r 


