# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

from matplotlib.pyplot import axis
import numpy as np
from random import sample

from numpy.core.fromnumeric import mean, size, std

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    #print(Points_2D.shape)
    coord_rows = Points_2D.shape[0]
    XYs = np.zeros((coord_rows*2, 11))
    UVs = []
    #Use this for loop to construct the matrix on the left
    for point in range(coord_rows):
        Ui = Points_2D[point, 0]
        Vi = Points_2D[point, 1]

        Xi = Points_3D[point, 0]
        Yi = Points_3D[point, 1]
        Zi = Points_3D[point, 2]

        XYs[point*2, 0] = Xi# E.g. X1 in row 1
        XYs[point*2, 1] = Yi# E.g. Y1 in row 1
        XYs[point*2, 2] = Zi# E.g. Z1 in row 1
        XYs[point*2, 3] = 1# E.g. 1 in row 1

        XYs[point*2+1, 4] = Xi# E.g. X1 in row 2
        XYs[point*2+1, 5] = Yi# E.g. Y1 in row 2
        XYs[point*2+1, 6] = Zi# E.g. Z1 in row 2
        XYs[point*2+1, 7] = 1# E.g. 1 in row 2

        XYs[point*2, 8] = -Ui * Xi# E.g. -u1*X1
        XYs[point*2, 9] = -Ui * Yi# E.g. -u1*Y1
        XYs[point*2, 10] = -Ui * Zi# E.g. -u1*Z1

        XYs[point*2+1, 8] = -Vi * Xi# E.g. -v1*X1
        XYs[point*2+1, 9] = -Vi * Yi# E.g. -v1*Y1
        XYs[point*2+1, 10] = -Vi * Zi# E.g. -v1*Z1
    
    UVs = np.reshape(Points_2D, (-1, 1))
    result = np.linalg.lstsq(XYs, UVs)[0]
    M = np.append(result, [1])
    M = M.reshape((3, 4))

    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################
    m4 = M[:, 3]
    Q = np.split(M, [3], axis=1)[0]

    Center = np.matmul(-np.linalg.inv(Q), m4)
    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    #Center = np.array([1,1,1]) 

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    ##################
    coord_rows = Points_a.shape[0]

    S_mat_a = np.eye(3)
    S_mat_b = np.eye(3)
    C_mat_a = np.eye(3)
    C_mat_b = np.eye(3)

    mean_a = np.mean(Points_a, axis=0)
    mean_b = np.mean(Points_b, axis=0)

    std_a = np.std(Points_a-mean_a)
    std_b = np.std(Points_b-mean_b)

    S_mat_a[0,0] = np.sqrt(2) / std_a
    S_mat_a[1,1] = np.sqrt(2) / std_a
    C_mat_a[0,2] = -mean_a[0]
    C_mat_a[1,2] = -mean_a[1]

    S_mat_b[0,0] = np.sqrt(2) / std_b
    S_mat_b[1,1] = np.sqrt(2) / std_b
    C_mat_b[0,2] = -mean_b[0]
    C_mat_b[1,2] = -mean_b[1]

    transform_T_a = np.matmul(S_mat_a, C_mat_a)
    transform_T_b = np.matmul(S_mat_b, C_mat_b)

    coord_a = np.concatenate((Points_a, np.ones(coord_rows).reshape(-1, 1)), axis=1)
    coord_b = np.concatenate((Points_b, np.ones(coord_rows).reshape(-1, 1)), axis=1)

    coord_a = np.matmul(transform_T_a, coord_a.T).T
    coord_b = np.matmul(transform_T_b, coord_b.T).T
    coord_a = np.tile(coord_a, 3)
    coord_b = np.repeat(coord_b, 3, axis=1)

    u,s,vh = np.linalg.svd(np.multiply(coord_a,coord_b))
    F_matrix = vh[-1].reshape((3,3))

    u,s,vh = np.linalg.svd(F_matrix)
    s[-1] = 0
    F_matrix = np.matmul(np.matmul(u, np.diagflat(s)), vh)
    F_matrix = np.matmul(np.matmul(transform_T_b.T, F_matrix), transform_T_a)
    #print(F_matrix)
    return F_matrix

# Takes h, w to handle boundary conditions
def apply_positional_noise(points, h, w, interval=3, ratio=0.2):
    """ 
    The goal of this function to randomly perturbe the percentage of points given 
    by ratio. This can be done by using numpy functions. Essentially, the given 
    ratio of points should have some number from [-interval, interval] added to
    the point. Make sure to account for the points not going over the image 
    boundary by using np.clip and the (h,w) of the image. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.clip

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] ( note that it is <x,y> )
            - desc: points for the image in an array
        h :: int 
            - desc: height of the image - for clipping the points between 0, h
        w :: int 
            - desc: width of the image - for clipping the points between 0, h
        interval :: int 
            - desc: this should be the range from which you decide how much to
            tweak each point. i.e if interval = 3, you should sample from [-3,3]
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will have some number from 
            [-interval, interval] added to the point. 
    """
    ##################
    # Your code here #
    ##################
    noises = np.random.rand(int(ratio * points.shape[0]), 2)
    noises = noises * interval + 1/5*interval - interval

    tweak = np.zeros((points.shape[0], 2))
    tweak[0:int(ratio * points.shape[0]), :] += noises
    np.random.shuffle(tweak)

    points += tweak
    points = np.clip(points, [0, 0], [h, w])
    return points

# Apply noise to the matches. 
def apply_matching_noise(points, ratio=0.2):
    """ 
    The goal of this function to randomly shuffle the percentage of points given 
    by ratio. This can be done by using numpy functions. 
    
    Key functions include but are not limited to:
        - np.random.rand
        - np.random.shuffle  

    Arugments:
        points :: numpy array 
            - shape: [num_points, 2] 
            - desc: points for the image in an array
        ratio :: float
            - desc: tells you how many of the points should be tweaked in this
            way. 0.2 means 20 percent of the points will be randomly shuffled.
    """
    ##################
    # Your code here #
    ##################
    noises = int(ratio * points.shape[0])
    tweak = np.zeros((points.shape[0], 2))
    tweak[0:noises, :] = 1
    np.random.shuffle(tweak)

    shuffle = np.copy(points[tweak == 1])
    np.random.shuffle(shuffle)
    points[tweak == 1] = shuffle

    return points


# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.
    s = 8 #This being the number of sampled points (Minimum number needed to fit the model)
    max_inlier_num = 0
    num_point = matches_a.shape[0]
    threshold = 0.009
    N = 1500 #This being the number of algorithm iterations

    for i in range(N):
        points = np.random.choice(num_point, size=s, replace = False)
        F_Matrix = estimate_fundamental_matrix(matches_a[points, :], matches_b[points, :])

        inlier_a = []
        inlier_b = []
        curr_inlier = 0
        for j in range(num_point):
            temp_match_a = np.append(matches_a[j, :], 1)
            temp_match_b = np.append(matches_b[j, :], 1)
            error_temp = np.matmul(temp_match_b.T, F_Matrix)
            error = np.matmul(error_temp, temp_match_a)
            error = abs(error)

            if error < threshold:
                curr_inlier += 1
                inlier_a.append(matches_a[j, :])
                inlier_b.append(matches_b[j, :])

        if max_inlier_num < curr_inlier:
            max_inlier_num = curr_inlier
            Best_Fmatrix = F_Matrix
            inliers_a = np.array(inlier_a)
            inliers_b = np.array(inlier_b)


    return Best_Fmatrix, inliers_a, inliers_b
