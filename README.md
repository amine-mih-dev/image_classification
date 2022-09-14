# image_classification
classifying images using transfer learning

This Notebook is broken down into multiple steps:
- load the image dataset (`Oxford Flowers 102 dataset`) from Tensorflow Hub and explore it, map the labels and creat a pipeline where images are resized and batched.
- Build and Train an image classifier on the dataset.
  - Loaded the MobileNet pre-trained network from TensorFlow Hub.
  - a new untrained feed-forward network as a classifier is defined
  - classifier is trained.
  - loss and accuracy during training and validation are displayed
  - make callbacks to save the best model 
  - training model is saved
- use the trained model to perform inference on flower images where we preprocess our new images to fit the model's input layer.
