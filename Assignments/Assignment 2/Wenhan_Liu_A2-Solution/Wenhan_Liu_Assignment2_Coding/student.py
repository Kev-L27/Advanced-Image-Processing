import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage.filters import gaussian,sobel_h, sobel_v
from scipy.ndimage import convolve

def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    xs = np.asarray([])
    ys = np.asarray([])
    Height = image.shape[0]
    Width = image.shape[1]

    blur_image = gaussian(image, sigma=0.1)

    sobelx = np.array(([-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]))
    sobely = np.array(([1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]))
    Sx = convolve(blur_image, sobelx)
    Sy = convolve(blur_image, sobely)

    Ix2 = gaussian(Sx**2, sigma=0.1)
    Ixy = gaussian((Sx * Sy), sigma=0.1)
    Iy2 = gaussian(Sy**2, sigma=0.1)
    theta = np.arctan2(Iy2, Ix2)

    NMwindow = np.zeros((Height, Width))
    for i in range(0, Height-16, 2):
      for j in range(0, Width-16, 2):
        Sx2 = Ix2[i:i+feature_width+1, j:j+feature_width+1]
        Sxy = Ixy[i:i+feature_width+1, j:j+feature_width+1]
        Sy2 = Iy2[i:i+feature_width+1, j:j+feature_width+1]

        Sx2 = Sx2.sum()
        Sxy = Sxy.sum()
        Sy2 = Sy2.sum()

        determinant = (Sx2*Sy2) - np.square(Sxy)
        trace = Sx2 + Sy2
        alpha = 0.06
        R = determinant - np.square(trace) * alpha
        threshold = 0.1
        if R > threshold:
          NMwindow[i+feature_width//2, j+feature_width//2] = R

    result = feature.peak_local_max(NMwindow, 3)    
    xs = result[:,1]
    ys = result[:,0]
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.
    details of why it helps are in the lecture content.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # This is a placeholder - replace this with your features!
    features = np.zeros((x.shape[0], 128))
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    #Blur the image with gaussian first
    gauss_image = gaussian(image, sigma=0.4)

    dx = sobel_v(gauss_image)
    dy = sobel_h(gauss_image)

    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan2(dy, dx)
    orientation[orientation < 0] += 2 * np.pi

    #Traverse the coordinates
    num_coord = len(x)
    for i in range(num_coord):
      #Find the 16x16 window and its corresponding magnitude and orientation
      window = gauss_image[y[i]-(feature_width//2):y[i]+(feature_width//2), x[i]-(feature_width//2):x[i]+(feature_width//2)]
      magnitude_wind = magnitude[y[i]-(feature_width//2):y[i]+(feature_width//2), x[i]-(feature_width//2):x[i]+(feature_width//2)]
      orientation_wind = orientation[y[i]-(feature_width//2):y[i]+(feature_width//2), x[i]-(feature_width//2):x[i]+(feature_width//2)]
      #Find all 4x4 window within the 16x16 window
      for j in range(4):
        for k in range(4):
          bin = np.zeros(8)
          #Find the 4x4 window and its corresponding magnitude and orientation
          small_window = window[j * 4:(j+1) * 4, k * 4:(k+1) * 4]
          small_mag = magnitude_wind[j * 4:(j+1) * 4, k * 4:(k+1) * 4]
          small_ori = orientation_wind[j * 4:(j+1) * 4, k * 4:(k+1) * 4]
          #8 directions, each 45 degree
          for degree in range(8):
            lower_bound = np.pi/4 * degree #45°* degree
            upper_bound = np.pi/4 * (degree+1)
            bin[degree] += np.sum(small_mag[np.all([small_ori >= lower_bound, small_ori < upper_bound], axis=0)])
          features[i, (4*j+k)*8:(4*j+k)*8+8] = bin
      #Normalization
      features[i,:] /= np.sqrt(np.sum(np.square(features[i,:])))
    return features

def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
             column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!

    feature_count = im1_features.shape[0] #Number of features

    matches = np.zeros((feature_count, 2))
    confidences = np.zeros(feature_count)

    #Traverse the features
    for i in range(feature_count):
      #Find the distance between feature in first image and all other features in second image 
      distance = np.sqrt(np.sum(np.square(im1_features[i,:]-im2_features), axis=1))
      #Find the smallest and second smallest distance and calculate NNDR
      index = np.argsort(distance)
      closest = distance[index[0]]
      sec_closest = distance[index[1]]
      NNDR = closest / sec_closest

      #Set the tolerance to 0.9 would give best results for the test images
      tolerance = 0.9
      if NNDR < tolerance:
        matches[i,:] = [i, index[0]]
        confidences[i] = 1/NNDR
    return matches, confidences