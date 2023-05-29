from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import numpy as np
import os


class L1DistCustom(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) 
       

def preprocess(file_path):
    
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (224,224))     
    img = img / 255.0
    return img

siameseModel = tf.keras.models.load_model('',
                                   custom_objects = {'L1DistCustom':L1DistCustom, 'BinaryCrossentropy':tf.keras.losses.BinaryCrossentropy}) #import model here
#siameseModel.summary()



def verify_images(model, detection_threshold, verification_threshold):
 
    validation_dir = '' # validation images directory
    input_dir = '' # input image directory here
    results = []
    for image in os.listdir(validation_dir):
        input_img = preprocess(os.path.join(input_dir, 'input_image.jpg'))
        validation_img = preprocess(os.path.join(validation_dir, image))
        
         
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
     
    detection = np.sum(np.array(results) > detection_threshold)
    
    
    verification = detection / len(os.listdir(validation_dir)) 
    verified = verification > verification_threshold
    
    return results, verified

rez, verif = verify_images(siameseModel, 0.5,0.5) # tune parameters
print(verif)
print("rezults")
print(rez)
if verif:
    print('t')
else:
    print('f')