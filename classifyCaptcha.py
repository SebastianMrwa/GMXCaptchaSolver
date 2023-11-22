# Author: Sebastian Mrwa, BSc
# This programme takes a Training-Captcha from the Training-Folder and does following steps:
# 1. Preprocessing like in the modelTraining
# 2. Using the CNN-Model to classify the Captcha

import numpy as np
from tensorflow.keras.models import load_model
import string
import os
import cv2

def fill_between_black_white_black(input_image):
    # Iterate through each column (x-axis)
    for x in range(input_image.shape[1]):
        column = input_image[:, x]

        for y in range(1, column.shape[0] - 1):
            # Check for a black-white-black sequence
            if column[y] == 255 and column[y - 1] == 0 and column[y + 1] == 0:
                # Fill the pixel in between as black
                input_image[y, x] = 0

    return input_image

def fill_between_black_white_black_horizontal(input_image):
    # Iterate through each row (y-axis)
    for y in range(input_image.shape[0]):
        row = input_image[y, :]

        for x in range(1, row.shape[0] - 1):
            # Check for a black-white-black sequence
            if row[x] == 255 and row[x - 1] == 0 and row[x + 1] == 0:
                # Fill the pixel in between as black
                input_image[y, x] = 0

    return input_image

# Function to remove lines with a height of 1 to 3 pixels
def remove_lines(image, adjacent_pixels=2):
    # Create a copy of the input image to work with
    result_image = np.copy(image)

    # Define the pattern to look for
    pattern = [255] + [0] * (adjacent_pixels - 1) + [255]

    # Iterate through the image and remove lines
    for x in range(result_image.shape[1]):
        for y in range(result_image.shape[0] - len(pattern) + 1):
            if all(result_image[y + i, x] == pattern[i] for i in range(len(pattern))):
                for i in range(len(pattern)):
                    result_image[y + i, x] = 255

    return result_image

def overwrite_left_pixels(image, threshold=30):
    result_image = np.copy(image)
    result_image[:, :threshold] = 255
    return result_image

def cut_upper_pixels(image, threshold=10):
    result_image = np.copy(image)
    result_image[:threshold, :] = 255
    return result_image

def split_contours(contours):
    new_contours = []

    if len(contours) == 5:
        new_contours = contours
    elif len(contours) == 1:
        # Split the single contour into 5 equal parts
        x, y, w, h = contours[0]
        split_width = w // 5
        for i in range(5):
            new_contours.append((x + i * split_width, y, split_width, h))
    else:
        # Sort contours by width in descending order
        contours = sorted(contours, key=lambda x: x[2], reverse=True)

        if len(contours) == 4:
            # Split the widest contour in half
            x, y, w, h = contours[0]
            split_width = w // 2
            new_contours.append((x, y, split_width, h))
            new_contours.append((x + w - split_width, y, split_width, h))
            new_contours.extend(contours[1:])
        elif len(contours) == 3:
            # Check if one contour is wider than the sum of the widths of the other 2
            if contours[0][2] > contours[1][2] + contours[2][2]:
                x, y, w, h = contours[0]
                split_width = w // 3
                for i in range(3):
                    new_contours.append((x + i * split_width, y, split_width, h))
                new_contours.append(contours[1])
                new_contours.append(contours[2])
            else:
                # Split the 2 widest contours in half
                for i in range(2):
                    x, y, w, h = contours[i]
                    split_width = w // 2
                    new_contours.append((x, y, split_width, h))
                    new_contours.append((x + w - split_width, y, split_width, h))
                new_contours.append(contours[2])
        elif len(contours) == 2:
            # Check if one contour is significantly wider than the other
            if contours[1][2] < contours[0][2] / 3:
                # Split the wider contour into 4 equal parts
                x, y, w, h = contours[0]
                split_width = w // 4
                for i in range(4):
                    new_contours.append((x + i * split_width, y, split_width, h))
                new_contours.append(contours[1])
            else:
                # Split the wider contour into 3 equal parts and the other one into 2
                x, y, w, h = contours[0]
                split_width_1 = w // 3
                new_contours.append((x, y, split_width_1, h))
                new_contours.append((x + split_width_1, y, split_width_1, h))
                new_contours.append((x + w - split_width_1, y, split_width_1, h))
                x, y, w, h = contours[1]
                split_width_2 = w // 2
                new_contours.append((x, y, split_width_2, h))
                new_contours.append((x + w - split_width_2, y, split_width_2, h))

    return new_contours

# Find the largest_index of the model output for classification
def find_largest_index(arr):
    # Flatten the 2D array
    flattened_arr = [item for sublist in arr for item in sublist]

    max_index = 0
    max_value = flattened_arr[0]

    for i in range(1, len(flattened_arr)):
        if flattened_arr[i] > max_value:
            max_value = flattened_arr[i]
            max_index = i

    return max_index

# Invert the dictionary to retreive the value
def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key

# Create a label mapping dictionary
label_mapping = {str(i): i for i in range(10)}  # Map numbers 0 to 9

# Map lowercase letters a to z to integers starting from 10
for i, char in enumerate(string.ascii_lowercase, start=10):
    label_mapping[char] = i

# Stores the extracted characters
characters = []

# Load the saved model
cnn_model = load_model('captchaCNN.keras')
folder_path = 'C:\Captchas'
png_file = '5p8fm.png'

image_path = os.path.join(folder_path, png_file)

# Step 1: Read the image
image = cv2.imread(image_path)

# Step 2: Convert to grayscale and threshold to make a binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Step 3: Fill pixels
filled_image = fill_between_black_white_black(binary_image)
filled_image_full = filled_image

# Step 4: Cut the left pixels
image_cut = overwrite_left_pixels(filled_image_full, 29)

# Step 5: Cut the upper pixels
image_cut_2 = cut_upper_pixels(image_cut, 10)

# Step 6: Remove lines (3px threshold) and second run (2px threshold)
line_removed_image_1 = remove_lines(image_cut_2, 3)
line_removed_image_2 = remove_lines(line_removed_image_1, 2)

# Step 7: Invert the image
inverted_cleaned_image = cv2.bitwise_not(line_removed_image_2)

# Step 8: Find contours on the inverted image
contours, _ = cv2.findContours(inverted_cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on height
min_contour_height = 15
filtered_contours = [c for c in contours if cv2.boundingRect(c)[3] > min_contour_height]

# Split contours logic
processed_filtered_contours = []

for c in filtered_contours:
    processed_filtered_contours.append(list(cv2.boundingRect(c)))
split_contours_var = split_contours(processed_filtered_contours)

# Sort the split_contours according to their appearance in the Captcha
sorted_contours = sorted(split_contours_var, key=lambda x: x[0])

# Safe characters
index = 0
for (x, y, w, h) in sorted_contours:
    # Create a mask for the current contour
    mask = np.zeros((35, 35), dtype=np.uint8)

    # Adjust width
    if w > 35:
        x = x + int((w-35)/2)
    elif w < 35:
        x = x - int((35-w)/2)

    # Adjust height
    if h > 35:
        y = y + int((h-35)/2)
    elif h < 35:
        y = y - int((35-h)/2)

    # Copy pixels from x to x+w and y to y+h to the mask
    if y+35 > 50:
        y = 50-35
    if x+35 > 200:
        x = 200-35
    mask[:35, :35] = inverted_cleaned_image[y:y+35, x:x+35]

    # Cut sides of characters that should not have been copied
    if w < 35:
        mask[:35, :int((35-w)/2)] = 0
        mask[:35, 35-int((35-w)/2):35] = 0

    # Save the resized and masked image
    characters.append(mask)

    index+=1

# Training the model
X = np.array(characters).reshape(len(characters), 35, 35, 1)

# Normalize pixel values to be between 0 and 1
X = X.astype('float32') / 255.0

predictions = []

for character in X:
    predictions.append(cnn_model.predict(character.reshape(1, 35, 35, 1)))

output = []

for pred in predictions:
    index = find_largest_index(pred)
    key = get_key_from_value(label_mapping, index)
    output.append(key)

captcha_prediction = ''.join(output)
print(captcha_prediction)