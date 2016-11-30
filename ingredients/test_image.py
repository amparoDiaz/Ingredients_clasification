# import the necessary packages
from keras.preprocessing import image as image_utils
import numpy as np
import argparse
import cv2
from keras import preprocessing
from keras.models import model_from_json
import pandas as pd
from keras import backend as K

def image_to_feature_vector(image_i, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image_i, size)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])

# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
cv2.imshow("Classification", orig)

	
# load the VGG16 network
print("[INFO] loading network...")
# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model/weight.h5")
print("Loaded model from disk")


# classify the image
print("[INFO] classifying image...")

feature = image_to_feature_vector(orig)
feature= np.expand_dims(feature, axis=0)
preds = model.predict(np.array(feature))
pred=preds
rounded = [round(x,1) for x in pred[0]]

ingredients = pd.read_csv('data/ingredients1.csv')
for (i, ingredient) in enumerate(ingredients.columns[0:10]):
    print i,ingredient, "=", rounded[i]

print(rounded)
# display the predictions to our screen
#print("ImageNet ID: {}, Label: {}".format(inID, label))
#cv2.putText(orig, "Label: {}".format(label), (10, 30),
#	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.waitKey(0)
