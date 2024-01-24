# %%
# import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications,models 
import numpy as np
import matplotlib.pyplot as plt
import os,datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
# %%
# load dataset
path = os.getcwd()
data_dir = os.path.join(path,'Concrete Crack Images for Classification')
# %%
BATCH_SIZE = 72
IMG_SIZE = (227,227)
# the batch for datasets
dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)
# %%
# inspect the data examples
class_names = dataset.class_names

plt.figure(figsize=(10,10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.grid('off')
# %%
# split train test val datasets
train_size = int(len(dataset)*0.7)
val_size = int(len(dataset)*0.2)
test_size = int(len(dataset)*0.1)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size).take(test_size)
# %%
# Convert the tensorflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE 

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)
val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)
test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)
# %%
# data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input
# %%
# construct the transfer learning pipeline ; pipeline: preprocess input > transfer learning model
# load the pretrained model using keras.applications
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.summary()
keras.utils.plot_model(base_model)
# %%
# freeze the entire feature extractor
base_model.trainable = False
base_model.summary()
# %%
# create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
# create the output layer with Dense layer
output_layer = layers.Dense(
    len(class_names),
    activation='softmax'
    )
# build the entire pipeline using functional API 
# Input
inputs = keras.Input(shape=IMG_SHAPE)
# data normalizaton - normalize the pixel
x = preprocess_input(inputs) # truediv & subtract come from this layer
# transfer learning feature extractor
x = base_model(x,training=False) # training=False because we don't want to entier base_model because of the freeze 
# classification layers
x = global_avg(x)
# here we can add the dropout layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
# build the model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
# %%
# compile the model
optimizer = optimizers.Adam(learning_rate=0.001)
loss =losses.SparseCategoricalCrossentropy()
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
    )
# %%
# prepare the callback object for model.fit()
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
    )
# perfoemance graph at tensorboard
path = os.getcwd()
logpath = os.path.join(path,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)
# # %%
# # evaluate the model with test data before training
# model.evaluate(test_dataset)
# %%
epochs = 15
hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping,tb]
)
# %%
fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='blue', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='lower right')
plt.show()
# %%
# evaluate model after training
model.evaluate(test_dataset)
# %%
# Model Deployment
# Retrieve the bath of sata from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch) 

# Identify the class  for prediction
prediction_index = np.argmax(predictions,axis=1)

# Display the result using matplotlib
label_map = {i:names for i,names in enumerate(class_names)}
prediction_list = [label_map[i] for i in prediction_index]
label_list = [label_map[i] for i in label_batch]

# Plot the image graph using matplotlib
plt.figure(figsize=(15,25))
for i in range(32):
    ax = plt.subplot(8,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"Prediction: {prediction_list[i]}\n Label: {label_list[i]}")
    plt.axis('off')
    plt.grid('off')
# %%
train_evaluation = model.evaluate(train_dataset)
val_evaluation = model.evaluate(val_dataset)
test_evaluation = model.evaluate(test_dataset)
# %%
# The evaluation of train test val for the model
print(f"Train Accuracy: {round((train_evaluation[1]*100), 2)}%")
print(f"Validation Accuracy: {round((val_evaluation[1]*100), 2)}%")
print(f"Test Accuracy: {round((test_evaluation[1]*100), 2)}%")
# %%
# Save the deep learning model
from tensorflow.keras.models import load_model
model.save(os.path.join('models', 'concrete_crack_classify.h5'))
# %%
