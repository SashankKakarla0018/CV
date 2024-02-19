from PIL import Image

image_path='/Users/shivashankar/Downloads/grain_image/8/DS-115_003'

# Load the TIFF image
tiff_image = Image.open(image_path+'.tif')  # Replace 'input.tif' with your TIFF image file path

# Save it as a JPEG image
tiff_image.save(image_path+'.jpg', 'JPEG')  # 'output.jpg' is the desired output file name


# Importing Image class from PIL module
from PIL import Image

# Opens a image in RGB mode
im = Image.open(image_path+'.jpg')

# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size

# Setting the points for cropped image
left = 0
top = 0
right = width
bottom = height-70

# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))


# Shows the image in image viewer
im1.save(image_path+'.jpg')

import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(image_path+'.jpg',1)
    
plt.imshow(image)
#plt.savefig("/Users/shivashankar/Downloads/original.jpg")

# Adjust the brightness and contrast
# Adjusts the brightness by adding 10 to each pixel value
brightness = 20
# Adjusts the contrast by scaling the pixel values by 2.3
contrast = 1

image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)


plt.title("Brightness & contrast")
plt.imshow(image2)
plt.show()

# Create the sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  
# Sharpen the image
image = cv2.filter2D(image2, -1, kernel)

plt.imshow(image)
#plt.savefig("/Users/shivashankar/Downloads/sharpened.jpg")

# Convert the cropped image to grayscale
gray_cropped = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray_cropped, (5, 5), 0)

# Apply thresholding to segment particles
threshold_value, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours of particles
contours, threshold_value = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate particle diameters and collect them in a list
particle_diameters = []
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2
    particle_diameters.append(diameter)

# Draw particle boundaries
cv2.drawContours(image2, contours, -1, (100, 0, 255), 2)  # Green color, thickness 2

average_diameter = np.mean(particle_diameters)
print("Average Diameter of Particles:", average_diameter,"(pixels)")





# Display the cropped image with particle boundaries
plt.imshow(image2)