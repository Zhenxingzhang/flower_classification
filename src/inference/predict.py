import sys
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import tensorflow as tf

RGB_MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)).astype(np.float32)

if __name__ == "__main__":
    print("Hello world")

    img = image.load_img(sys.argv[1], target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, mode='caffe')
    # img = img - RGB_MEAN_PIXELS
    # img = tf.reverse(img, axis=[-1])

    model = ResNet50(weights='imagenet')
    input_tensor = model.input
    output_tensor = model.output

    sess = K.get_session()
    keras_output = sess.run(output_tensor, {input_tensor: img})
    print(decode_predictions(keras_output, top=3)[0])
