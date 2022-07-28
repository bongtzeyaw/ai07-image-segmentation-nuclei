import numpy as np
import matplotlib.pyplot as plt

from scipy import io
import glob, os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split

SEED = 1
IMG_SIZE = (227,227)
BATCH_SIZE = 32

# Load data
images = []
masks = []

root_path = r"C:\Users\tzeyawbong\Documents\shrdc\git\files2"

image_dir = os.path.join(root_path, 'inputs')
for image_file in os.listdir(image_dir):
  img = cv2.imread(os.path.join(image_dir,image_file))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = cv2.resize(img,(128,128))
  images.append(img)

mask_dir = os.path.join(root_path,'masks')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir, mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)

images_np = np.array(images)
masks_np = np.array(masks)

# Display data
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_np[i])
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))

for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_np[i])
    plt.axis('off')
plt.show()

masks_np_exp = np.expand_dims(masks_np, axis=-1)
converted_masks = np.round(masks_np_exp/255).astype(np.int64)
converted_images = images_np/255.0

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))

class Augment(layers.Layer):
  def __init__(self,seed=SEED):
    super().__init__()
    self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=SEED)
    self.augment_labels = layers.RandomFlip(mode='horizontal',seed=SEED)
  
  def call(self,inputs,labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs,labels

# Create model
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_expand_relu',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512,3),
    pix2pix.upsample(256,3),
    pix2pix.upsample(128,3),
    pix2pix.upsample(64,3),
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
    last = layers.Conv2DTranspose(
        filters = output_channels,kernel_size=3,strides=2,padding='same'
    )
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)

loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS

model.fit(train_batches, validation_data=test_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[...,tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None,num=1):
  if dataset:
    for image,mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0],mask[0],create_mask(pred_mask)])
  else:
    display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions(test_batches,3)

# Evaluate model
model.evaluate(test_batches)