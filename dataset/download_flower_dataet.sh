#!/bin/bash

# Set the URL and filename of the ZIP file to download
ZIP_URL="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ZIP_FILENAME="flower_photos.tgz"

# Download the ZIP file
wget $ZIP_URL -O $ZIP_FILENAME

# Unzip the file
tar -xvzf $ZIP_FILENAME

# Remove the ZIP file
rm $ZIP_FILENAME
