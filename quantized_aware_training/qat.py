import tensorflow as tf
import os


# run the download_flower_dataset.sh in dataset folder first

flowers_dir = "/home/walter/git/edgetpu/dataset/flower_photos"

IMAGE_SIZE = 224
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    flowers_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    flowers_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

# image_batch, label_batch = next(val_generator)
# print(image_batch.shape)
# print(label_batch.shape)

# save label file
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

label_file_savepath = os.path.join(flowers_dir, 'flower_labels.txt')
with open(label_file_savepath, 'w') as f:
    f.write(labels)