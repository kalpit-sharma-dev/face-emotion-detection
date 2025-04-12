import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Constants
img_height, img_width = 48, 48
channels = 1
batch_size = 64
time_steps = 6
chunk_size = 8  # 6 chunks of width 8 to match 48 width

# Paths
train_dir = 'train'
test_dir = 'test'

# Image Generators (no augmentation for now)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

# Helper: Reshape batches to (samples, time_steps, height, chunk_size, 1)
def reshape_batch(batch_x):
    return batch_x.reshape(-1, time_steps, img_height, chunk_size, 1)

# Build CNN + RNN Model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(time_steps, img_height, chunk_size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(SimpleRNN(64))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Custom training loop to reshape batches for CNN + RNN
epochs = 10
steps_per_epoch = train_generator.samples // batch_size
validation_steps = test_generator.samples // batch_size

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    for step in range(steps_per_epoch):
        batch_x, batch_y = next(train_generator)
        batch_x_rnn = reshape_batch(batch_x)
        model.train_on_batch(batch_x_rnn, batch_y)

    # Validation
    val_accuracy = []
    val_loss = []
    for _ in range(validation_steps):
        val_x, val_y = next(test_generator)
        val_x_rnn = reshape_batch(val_x)
        loss, acc = model.evaluate(val_x_rnn, val_y, verbose=0)
        val_accuracy.append(acc)
        val_loss.append(loss)
    print(f"Validation Loss: {np.mean(val_loss):.4f}, Accuracy: {np.mean(val_accuracy):.4f}")

# Save model
model.save("cnn_rnn_model_from_dir.h5")
