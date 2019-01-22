import tensorflow as tf
from tensorflow.keras import layers
from read_slim_flowers import _parse_function

if __name__ == '__main__':
    EPOCHS = 10

    train_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_train_00000-of-00005.tfrecord",
        "/data/flowers/flowers_train_00001-of-00005.tfrecord",
        "/data/flowers/flowers_train_00002-of-00005.tfrecord",
        "/data/flowers/flowers_train_00003-of-00005.tfrecord",
        "/data/flowers/flowers_train_00004-of-00005.tfrecord"])
    train_dataset = train_dataset.map(_parse_function).batch(32).shuffle(1000).repeat()
    print(train_dataset.output_types, train_dataset.output_shapes)

    val_dataset = tf.data.TFRecordDataset([
        "/data/flowers/flowers_validation_00000-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00001-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00002-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00003-of-00005.tfrecord",
        "/data/flowers/flowers_validation_00004-of-00005.tfrecord"
    ]).map(_parse_function).batch(32).repeat()

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    batch_image, batch_label = iterator.get_next()
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(val_dataset)

    # define Keras model
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=5, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=10, steps_per_epoch=30,
              validation_data=val_dataset,
              validation_steps=3)
