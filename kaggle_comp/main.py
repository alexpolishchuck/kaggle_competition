#TODO:
# 1) Load data
# 2) Process images
# 3) Build model
# 4) Save predictions to csv
import os
import csv
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

CSV_TRAIN_FILE = "ukraine-ml-bootcamp-2023/train.csv"
MAX_CLASS_VALUE = 5.0
TRAIN_FOLDER = "ukraine-ml-bootcamp-2023/images/train_images"
TEST_FOLDER = "ukraine-ml-bootcamp-2023/images/test_images"
RESIZE_VALUE = 256
RGB = 3
TOTAL_SIZE_OF_PHOTOS = 2360

def load_csv_data():

    names = []
    classes = []

    with open(CSV_TRAIN_FILE) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            names.append(row[0])
            classes.append(float(row[1]))

    return dict(zip(names, classes))


def process_image(path_to_folder):

    min_size_x = -1
    min_size_y = -1
    avg_x = 0
    avg_y = 0


    for image_name in os.listdir(path_to_folder):
        image = Image.open(path_to_folder + "/" + image_name)
        x, y = image.size
        if (x < min_size_x) or (min_size_x == -1):
            min_size_x = x
        if (y < min_size_y) or (min_size_y == -1):
            min_size_y = y

        avg_x += x
        avg_y += y

    avg_x /= TOTAL_SIZE_OF_PHOTOS
    avg_y /= TOTAL_SIZE_OF_PHOTOS
    print("minx = " , min_size_x)
    print('miny = ' , min_size_y)
    print("avgx = " , avg_x)
    print('avgy = ' , avg_y)

def load_images(path_to_folder):
    names = []
    images = []

    for image_name in os.listdir(path_to_folder):
        image = Image.open(path_to_folder + "/" + image_name)
        # plt.imshow(image)
        # plt.show()
        image = image.convert('L')
        image = image.resize((RESIZE_VALUE, RESIZE_VALUE))
        image = np.asarray(image)

        if np.shape(image) != (RESIZE_VALUE, RESIZE_VALUE, RGB):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        images.append(image)
        names.append(image_name)

    images = np.array(images).astype(float) / 255.0
    names = np.array(names)

    return names, images

def prepare_data():
    name_to_class_map = load_csv_data()
    # TODO: augment data
    names, images = load_images(TRAIN_FOLDER)
    output = []

    for name in names:
        class_val = name_to_class_map[name]
        output.append(class_val)

    output = np.array(output)
    train_X, validate_X, train_Y, validate_Y = train_test_split(images, output, test_size=0.2, random_state=1)

    return train_X, validate_X, train_Y, validate_Y


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                               input_shape=(RESIZE_VALUE, RESIZE_VALUE, RGB)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-7),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def find_lr(train_X, validate_X, train_Y, validate_Y ):
    model = create_model()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10**(epoch/5))
    history = model.fit(train_X,
              train_Y,
              epochs=20,
              validation_data=(validate_X, validate_Y),
              callbacks=[lr_schedule])
    lrs = 1e-6 * (10**(np.arange(20)/5))
    # Set the figure size
    plt.figure(figsize=(10, 6))
    # Set the grid
    plt.grid(True)
    # Plot the loss in log scale
    plt.semilogx(lrs, history.history["loss"])
    # Increase the tickmarks size
    plt.tick_params('both', length=10, width=1, which='both')
    # Set the plot boundaries
    plt.axis([1e-6, 1e-3, 0, 10])
    plt.show()


def fill_results(model):
    with open("results.csv", 'w') as file:
        names, images = load_images(TEST_FOLDER)

        writer = csv.writer(file)
        writer.writerow(["image_id", "class_6"])

        res = model.predict(images)

        size = len(res)

        for i in range(0,size):
            writer.writerow([names[i], res[i]])


if __name__ == '__main__':
    process_image(TRAIN_FOLDER)
    train_X, validate_X, train_Y, validate_Y = prepare_data()
    model = create_model()
    model.summary()
    # find_lr(train_X, validate_X, train_Y, validate_Y)
    history = model.fit(train_X, train_Y, epochs=400, validation_data=(validate_X, validate_Y))
    fill_results(model)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()