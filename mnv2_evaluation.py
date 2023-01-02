"""
TODO: use evaluation function
"""
import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from get_labelmap import get_labelmap

# setup path
base_dir = "/home/pymi/dataset/ILSVRC_eval"
val_dir = os.path.join(base_dir, "Data/CLS-LOC/val")
label_filepath = os.path.join(base_dir, "LOC_val_solution.csv")
labelmap = get_labelmap(label_filepath)
# setup GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# create model: mnv2 as example
model = MobileNetV2(weights='imagenet')
print(model.summary())


top5_collect = []
top1_collect = []
# loop image 
val_images = os.listdir(val_dir)
start = time.time()
for val_image in tqdm(val_images):
    key = val_image.split('.')[0]
    label = labelmap[key]
    val_image_path = os.path.join(val_dir, val_image)
    img = image.load_img(val_image_path, target_size=(224, 224), keep_aspect_ratio=True, interpolation='bicubic')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x, verbose=0), top=5)[0]
    top5_result = [result[0] for result in preds]
    top1_result = preds[0][0]
    if label in top5_result:
        top5_collect.append(1)
    else:
        top5_collect.append(0)
    if label == top1_result:
        top1_collect.append(1)
    else:
        top1_collect.append(0)
end = time.time()
print("top5 accuracy: {}".format(np.mean(top5_collect)))
print("top1 accuracy: {}".format(np.mean(top1_collect)))
print("total evaluation time: {:.2f} s".format(end - start))