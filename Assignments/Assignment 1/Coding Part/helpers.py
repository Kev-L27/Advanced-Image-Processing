# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt

def my_imfilter(image, filter):
  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c)
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  filtered_image = np.asarray([0])

  #Use this if statement to check if the filter has valid dimension
  if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
    raise Exception('The filter must have odd dimension. Provided filter is invalid.')
  
  #Convert to grayscale image if input was an RGB image
  if len(image.shape) == 3:
    img_gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    channels = 3
  else:
    channels = 1
  
  #Pad the input image with corresponding amount of zeros in order to for the filtered image to have the same size with input image
  filterY = filter.shape[0]
  filterY_mid = filterY//2
  filterX = filter.shape[1]
  filterX_mid = filterX//2

  if channels == 3:#rgb image
    shape = image.shape[0:2] # shape[0] is the y and shape[1] is the x

    padded_img_r = np.zeros((shape[0] + 2*filterY_mid, shape[1] + 2*filterX_mid))
    padded_img_g = np.zeros((shape[0] + 2*filterY_mid, shape[1] + 2*filterX_mid))
    padded_img_b = np.zeros((shape[0] + 2*filterY_mid, shape[1] + 2*filterX_mid))
    #Do filter operations on these 3 channels and combine them later
    padded_img_r[filterY_mid:filterY_mid+shape[0], filterX_mid:filterX_mid+shape[1]] = image[:,:,0]
    padded_img_g[filterY_mid:filterY_mid+shape[0], filterX_mid:filterX_mid+shape[1]] = image[:,:,1]
    padded_img_b[filterY_mid:filterY_mid+shape[0], filterX_mid:filterX_mid+shape[1]] = image[:,:,2]

    #Actual Convolution Process
    filtered_image_r = np.zeros_like(img_gray)
    filtered_image_g = np.zeros_like(img_gray)
    filtered_image_b = np.zeros_like(img_gray)

    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        filtered_image_r[y][x] += (filter*padded_img_r[y:y+filterY, x:x+filterX]).sum()
        filtered_image_g[y][x] += (filter*padded_img_g[y:y+filterY, x:x+filterX]).sum()
        filtered_image_b[y][x] += (filter*padded_img_b[y:y+filterY, x:x+filterX]).sum()

    filtered_image = np.stack((filtered_image_r, filtered_image_g, filtered_image_b), axis=2)
  else:
    shape = image.shape
    padded_img = np.zeros((shape[0] + 2*filterY_mid, shape[1] + 2*filterX_mid))
    padded_img[filterY_mid:filterY_mid+shape[0], filterX_mid:filterX_mid+shape[1]] = image

    #Actual Convolution Process
    filtered_image = np.zeros_like(image)

    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        filtered_image[y][x] = (filter*padded_img[y:y+filterY, x:x+filterX]).sum()

  return filtered_image

def gen_hybrid_image(image1, image2, cutoff_frequency):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies. #Can be seen from further distance
   - image2 -> The image from which to take the high frequencies. #Can be seen from close distance
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
  s, k = cutoff_frequency, cutoff_frequency*2
  probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
  kernel = np.outer(probs, probs)# This is the Gaussian Filter
  
  #print("starting low")
  # Your code here:
  low_frequencies = my_imfilter(image1, kernel) # Replace with your implementation
  #print(low_frequencies)

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  #print("starting high")
  high_frequencies = image2 - my_imfilter(image2, kernel) # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  #print("starting hybrid")
  hybrid_image = low_frequencies + high_frequencies # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  #print(cur_image.shape)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    #print(output.shape)
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
    #print(cur_image.shape)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    #print(pad.shape)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
