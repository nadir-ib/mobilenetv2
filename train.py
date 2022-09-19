# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# keras imports
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
# from keras.utils import to_categorical
# from tensorflow.keras.utils import to_categorical
# from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
# other imports
import json
import datetime
import time

from utils import generate_batches, generate_batches_with_augmentation, create_folders
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load the user configs
with open('conf.json') as f:    
  config = json.load(f)

# config variables
weights     = config["weights"]
train_path    = config["train_path"]
test_path     = config["test_path"]
model_path    = config["model_path"]
batch_size    = config["batch_size"]
epochs        = config["epochs"]
classes       = config["classes"]
logs_data     = config["logs_data"]
validation_split   = config["validation_split"]
data_augmentation  = config["data_augmentation"]
epochs_after_unfreeze = config["epochs_after_unfreeze"]
checkpoint_period = config["checkpoint_period"]
checkpoint_period_after_unfreeze = config["checkpoint_period_after_unfreeze"]
IMAGE_SIZE = (224, 224)
create_folders(model_path, logs_data)

# folder = '/home/nadir/liveness/mobilenet-v2-custom-dataset/imgs'
# def get_classes_list(folder):
#     sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
#     return sub_folders


def generators(shape, preprocessing):
    '''Create the training and validation datasets for
    a given image shape.
    '''
    train_datagen = ImageDataGenerator(preprocessing_function=preprocessing)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing,)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing,)


    train_generator = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                        shuffle=True) #, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_path, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                    shuffle=False)# , class_mode='binary')

    val_generator = val_datagen.flow_from_directory(test_path, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                    shuffle=False)# , class_mode='binary')

    return train_generator, test_generator, val_generator

train_dataset, test_dataset, val_dataset = generators((224, 224), preprocessing=preprocess_input)


# create model
if weights=="imagenet":
  base_model = MobileNetV2(include_top=False, weights=weights, 
                            input_tensor=Input(shape=(224, 224, 3)), input_shape=(224, 224, 3))
  top_layers = base_model.output
  top_layers = GlobalAveragePooling2D()(top_layers)
  top_layers = Dense(1024, activation='relu')(top_layers)
  predictions = Dense(classes, activation='softmax')(top_layers)
  model = Model(inputs=base_model.input, outputs=predictions)
elif weights=="":
  base_model = MobileNetV2(include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)), input_shape=(224, 224, 3))
  top_layers = base_model.output
  top_layers = GlobalAveragePooling2D()(top_layers)
  top_layers = Dense(1024, activation='relu')(top_layers)
  predictions = Dense(classes, activation='softmax')(top_layers)
  model = Model(inputs=base_model.input, outputs=predictions)
else:
  model = load_model(weights)
print ("[INFO] successfully loaded base model and model...")

# create callbacks
checkpoint = ModelCheckpoint("logs/weights.h5", monitor='loss', save_best_only=True, period=checkpoint_period)

# start time
start = time.time()

print ("Freezing the base layers. Unfreeze the top 2 layers...")
for layer in model.layers[:-3]:
    layer.trainable = False


opt = optimizers.Adam(lr=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', ])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print("Start training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    workers=10,
    epochs=epochs,
    callbacks=[checkpoint, ]
)

# model.fit_generator(generate_batches(train_path, batch_size), epochs=epochs,
#         steps_per_epoch=samples//batch_size, verbose=1, callbacks=[checkpoint])

print ("Saving...")
model.save(model_path + "/save_model_stage1.h5") 

# print ("Visualization...")
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# if epochs_after_unfreeze > 0:
#   print ("Unfreezing all layers...")
#   for i in range(len(model.layers)):
#     model.layers[i].trainable = True
#   model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
#   print ("Start training - phase 2...")
#   checkpoint = ModelCheckpoint("logs/weights.h5", monitor='loss', save_best_only=True, period=checkpoint_period_after_unfreeze)
#   if data_augmentation:
#     model.fit_generator(
#           generate_batches_with_augmentation(train_path, batch_size, validation_split, augmented_data),
#           verbose=1, epochs=epochs, callbacks=[checkpoint])
#   else:
#     model.fit_generator(generate_batches(train_path, batch_size), epochs=epochs_after_unfreeze,
#           steps_per_epoch=samples//batch_size, verbose=1, callbacks=[checkpoint])
#
#   print ("Saving...")
#   model.save(model_path + "/save_model_stage2.h5")
#
# # end time
# end = time.time()
# print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
# print ("[STATUS] total duration: {}".format(end - start))
