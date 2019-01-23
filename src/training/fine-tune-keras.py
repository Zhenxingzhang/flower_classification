import tensorflow as tf
from tensorflow.python.keras.applications import resnet50
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras import backend as K


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

    image = tf.image.resize_image_with_pad(image, 224, 224)

    # Images need to have the same dimensions for feeding the network
    # this did not include data augmentation.
    image = resnet50.preprocess_input(image)

    return image, label


if __name__=="__main__":



    train_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_train_00000-of-00005.tfrecord",
        "/data/flowers/flowers_train_00001-of-00005.tfrecord",
        "/data/flowers/flowers_train_00002-of-00005.tfrecord",
        "/data/flowers/flowers_train_00003-of-00005.tfrecord",
        "/data/flowers/flowers_train_00004-of-00005.tfrecord"])
    train_dataset = train_dataset.map(_parse_function).shuffle(1000).batch(32).repeat()

    val_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_validation_00000-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00001-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00002-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00003-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00004-of-00005.tfrecord"
    ]).map(_parse_function).batch(32)

    # iterator = train_dataset.make_initializable_iterator()
    # (imgs, labels) = iterator.get_next()
    #
    # sess = tf.keras.backend.get_session()
    # sess.run(iterator.initializer)
    # batch = sess.run([imgs, labels])
    # print(imgs.shape)

    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model on the new data for a few epochs
    model.fit(train_dataset, epochs=10, steps_per_epoch=30
              # validation_data=val_dataset, validation_steps=1
              )

    print(tf.get_default_graph().get_tensor_by_name("Const:0"))
    print(tf.get_default_graph().get_tensor_by_name("FlatMapDataset_1:0"))

    # for n in tf.get_default_graph().as_graph_def().node:
    #     print(n.name)

