import argparse
import json 

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image

print(tf.__version__)

my_parser = argparse.ArgumentParser(description='enter image name')

my_parser.add_argument('image', action='store', type=str, help=' path to the image')
my_parser.add_argument('model', action='store', type=str, help='the path to the model')
my_parser.add_argument('--top_k', default= 5, action='store', type=int, help='return the top k most likely classes')
my_parser.add_argument('--category_names', default='./label_map.json', action='store', type=str, help ='JSON file mapping labels')

args = my_parser.parse_args()
#image_path = args.image
top_k = args.top_k


def process_image(image):
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    image = image.reshape((224, 224, 3))
    image = image.astype('float32')
    return image

def predict(image_path, model, k_classes):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = tf.expand_dims(img, axis = 0)
    probs = model.predict(img)
    values, indices = tf.nn.top_k(probs,k = k_classes)
    probs = list(values.numpy()[0])
    classes = list(indices.numpy()[0])
    return probs, classes

with open(args.category_names, "r") as file:
    mapping = json.load(file)
    
loaded_model = tf.keras.models.load_model(args.model, custom_objects = {'KerasLayer':hub.KerasLayer}, compile=False)

print(f"\n Top {top_k} Classes\n")
probs, labels = predict(args.image, loaded_model, top_k)
for prob, label in list(zip(probs, labels)):
    print('Label:', label)
    print('Class name:', mapping[str(label+1)].title())
    print('probability:', prob)




