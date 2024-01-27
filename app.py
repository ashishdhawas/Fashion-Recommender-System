import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# import ResNet50 model with imagenet dataset trained weights with excluding top layer
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D() # add top layer as maxpooling2D layer
])

# print(model.summary())

# create function for extracting feature for the images using ResNet50 model
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224)) # load image
    img_array = image.img_to_array(img) # covert image to array
    # expand the dimention of array beacause model expected batches images
    expanded_img_array = np.expand_dims(img_array, axis=0) 
    # The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# stored the file name of the images
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# stored the extracted feature from the model
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

# create the pickle file for extracted features
pickle.dump(feature_list,open('embeddings.pkl','wb'))
# create the pickle file for the images
pickle.dump(filenames,open('filenames.pkl','wb'))