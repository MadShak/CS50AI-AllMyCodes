import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Constants – feel free to tweak these while experimenting
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load images and labels
    images, labels = load_data(sys.argv[1])

    # Split into train and test sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Returns a tuple (images, labels):
        - images: list of numpy arrays, each representing a resized image.
        - labels: list of integers, corresponding category for each image.
    """
    images = []
    labels = []

    # data_dir contains subfolders 0, 1, 2, ..., 42
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            # Build full file path
            file_path = os.path.join(category_path, filename)

            # Read image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                # Skip files that can't be read (e.g. hidden files)
                continue

            # Resize to the fixed dimensions
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Append to our lists
            images.append(img_resized)
            labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model.

    Architecture:
        - Conv2D + ReLU + MaxPooling
        - Conv2D + ReLU + MaxPooling
        - Flatten
        - Dense hidden layer with Dropout
        - Output layer with softmax (43 categories)
    """
    model = tf.keras.models.Sequential([

        # First convolutional layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer – deeper, more filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten before feeding into dense layers
        tf.keras.layers.Flatten(),

        # Hidden dense layer with dropout to reduce overfitting
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer: one unit per category, softmax activation
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()
