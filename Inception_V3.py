import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score


train_data_dir = '/content/dataset/Deep_Dive_MLathon_24_dataset/Train_data'
validation_data_dir = '/content/dataset/Deep_Dive_MLathon_24_dataset/Validation_data'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),  # InceptionV3 requires input size of (299, 299)
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(299, 299),  # InceptionV3 requires input size of (299, 299)
    batch_size=32,
    class_mode='categorical'
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Unfreeze some layers for fine-tuning
for layer in base_model.layers:
    layer.trainable = False

# Add dropout for regularization
x = base_model.output
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout
x = BatchNormalization()(x)
predictions = Dense(10, activation='softmax')(x)

# Reconstruct model with new layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model with a lower learning rate and a learning rate scheduler
initial_learning_rate = 0.0001
epochs = 35

lr_schedule = LearningRateScheduler(lambda epoch: initial_learning_rate * 0.9 ** epoch)


def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    actual_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_score = 2 * ((precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon()))
    return f1_score


model.compile(optimizer=Adam(lr=initial_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', precision, recall, f1_score])

# Summary of the model
model.summary()

# Train the model with improved settings
history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs, callbacks=[lr_schedule])
