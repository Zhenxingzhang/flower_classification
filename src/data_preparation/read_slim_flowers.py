import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append("/data/slim/models/research/slim/")
from preprocessing import vgg_preprocessing

from tensorflow import keras
from tensorflow.keras import layers

def _parse_function(example_proto):
    features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"])
    width = tf.cast(parsed_features["image/width"], tf.int32)
    height = tf.cast(parsed_features["image/height"], tf.int32)
    label = tf.cast(parsed_features["image/class/label"], tf.int32)

    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_image_with_pad(image, 50, 50)

    # Images need to have the same dimensions for feeding the network
    image = vgg_preprocessing.preprocess_image(image, 32, 32)

    return image, label


if __name__ == '__main__':
    EPOCHS = 10

    train_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_train_00000-of-00005.tfrecord",
        "/data/flowers/flowers_train_00001-of-00005.tfrecord",
        "/data/flowers/flowers_train_00002-of-00005.tfrecord",
        "/data/flowers/flowers_train_00003-of-00005.tfrecord",
        "/data/flowers/flowers_train_00004-of-00005.tfrecord"])
    train_dataset = train_dataset.map(_parse_function).batch(32).shuffle(1000)
    print(train_dataset.output_types, train_dataset.output_shapes)

    val_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_validation_00000-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00001-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00002-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00003-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00004-of-00005.tfrecord"
    ]).map(_parse_function).batch(32)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    batch_image, batch_label = iterator.get_next()
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(val_dataset)

    # define Keras model
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=5, activation='softmax'))
    logits = model(batch_image)

    predictions = tf.to_int32(tf.argmax(logits, 1))
    tf.losses.sparse_softmax_cross_entropy(labels=batch_label, logits=logits)
    total_loss = tf.losses.get_total_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    # use this op to train the whole network
    full_train_op = optimizer.minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # initialise iterator with train data
        for i in range(EPOCHS):
            print('Starting training epoch %d / %d' % (i + 1, EPOCHS))
            sess.run(training_init_op)
            while True:
                try:
                    # train on one batch of data
                    _, tot_loss = sess.run([full_train_op, total_loss])
                except tf.errors.OutOfRangeError:
                    break
            print("Iter: {}, Loss: {:.4f}".format(i, tot_loss))

        # initialise iterator with test data
        sess.run(validation_init_op)
        print('Test Loss: {:4f}'.format(sess.run(total_loss)))

