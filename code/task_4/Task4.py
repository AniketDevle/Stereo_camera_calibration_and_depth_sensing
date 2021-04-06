import numpy as np
import cv2
import os.path
import math
from matplotlib import pyplot as plt

def get_images() :
    """This function will read the images from the task_2 folder and return them

    return format: image_one , image_two
    """

    left_img = cv2.imread('../../images/task_3_and_4/left_6.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('../../images/task_3_and_4/right_6.png', cv2.IMREAD_GRAYSCALE)
    return left_img, right_img

def read_arrays(path, keys):
    """Reads arrays from an XML file using cv2.FileStorage.
    Args:
        path: Path to the XML file
        keys: Keys to access the arrays
    """
    reader = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    arrays = {}

    # TODO Raise exception for unavailable keys
    for key in keys:
        arrays[key] = reader.getNode(key).mat()
    reader.release()

    return arrays

def get_camera_intrinsics():

    """This function will load the camera matrix and distortion coefficient of both camera that we calculated in task one from the csv file

    return format: Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right

    """


    left_camera_intrinsics = read_arrays(
        os.path.join('../../parameters/', 'left_camera_intrinsics.xml'),
        ['Camera_matrix_left', 'distortion_coef_left'])
    right_camera_intrinsics = read_arrays(
        os.path.join('../../parameters/', 'right_camera_intrinsics.xml'),
        ['Camera_matrix_right', 'distortion_coef_right'])

    return left_camera_intrinsics['Camera_matrix_left'], left_camera_intrinsics['distortion_coef_left'], \
           right_camera_intrinsics['Camera_matrix_right'], right_camera_intrinsics['distortion_coef_right']

def get_stereo_calibration_parameters():
    """This function will load the Rotation matrix, Translation of camera 2 with respect to camera one and essential
    and fundamental matrix

    return format:
        R : rotation matrix of second camera with respect to first camera
        T : translation of second camera with respect to first camera
        E : Essential matrix
        F : Fundamental Matrix

    """
    Stereo_calibration_parameters = read_arrays(os.path.join('../../parameters/', 'stereo_calibration.xml'),
                                                ['R', 'T', 'E', 'F'])

    return Stereo_calibration_parameters['R'], Stereo_calibration_parameters['T'], Stereo_calibration_parameters['E'], \
           Stereo_calibration_parameters['F']

def get_stereo_rectification_parameters():
    """This function will load the rotation of left and right camera, projection matrix (new camera matrix) of left and right
    and disparity to depth mapping

    return format:
        R_l : rotation left
        R_r : rotation right
        P_l : projection matrix left
        P_r : projection matrix right
        Q : disparity to depth mapping

    """
    Stereo_rectification_parameters = read_arrays(os.path.join('../../parameters/', 'stereo_rectification.xml'),
                                                  ['R_l', 'R_r', 'P_l', 'P_r', 'Q'])

    return Stereo_rectification_parameters['R_l'], Stereo_rectification_parameters['R_r'], \
           Stereo_rectification_parameters['P_l'], Stereo_rectification_parameters['P_r'], \
           Stereo_rectification_parameters['Q']


def disparity_map(undistorted_img_left, undistorted_img_right):
    """
    wsize = 31
    max_disp = 128
    sigma = 1.5
    lmbda = 8000.0
    left_matcher = cv2.StereoBM_create(max_disp, wsize);
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher);
    left_disp = left_matcher.compute(left_image, right_image);
    right_disp = right_matcher.compute(right_image, left_image);

    # Now create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher);
    wls_filter.setLambda(lmbda);
    wls_filter.setSigmaColor(sigma);
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp);
    """
    win_size = 7
    min_disp = 0
    max_disp = 64
    num_disp = max_disp - min_disp

    #create block matching object.
    matcher = cv2.StereoSGBM_create(
        minDisparity= min_disp,
        numDisparities= num_disp,
        blockSize = win_size,
        preFilterCap= 63,
        uniquenessRatio= 15,
        speckleWindowSize=10,
        speckleRange= 1,
        disp12MaxDiff= 20,
        P1= 8*3*win_size**2,
        P2 = 32*3*win_size**2,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    left_matcher = matcher
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    l = 70000
    s = 1.2

    disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    disparity_filter.setLambda(l)
    disparity_filter.setSigmaColor(s)

    d_l = left_matcher.compute(undistorted_img_left,undistorted_img_right)
    d_r = right_matcher.compute(undistorted_img_right,undistorted_img_left)

    d_l = np.int16(d_l)
    d_r = np.int16(d_r)

    d_filter = disparity_filter.filter(d_l,undistorted_img_left,None,d_r)

    #plt.imshow(d_filter, cmap='gray')
    #plt.show()
    cv2.imwrite('../../output/task_4/Disparity_map_scene_6.png' , d_filter)


    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    #disparity = stereo.compute(left_img, right_img)
    #plt.imshow(filtered_disp, 'gray')
    #plt.show()

    return d_filter


if __name__ == '__main__':
    # Get left and right images in grayscale
    left_img, right_img = get_images()

    # Get intrinsic parameters
    Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right = get_camera_intrinsics()

    # Get Stereo calibration parameters
    R, T, E, F = get_stereo_calibration_parameters()

    # get Stereo rectification parameters
    R_l, R_r, P_l, P_r, Q = get_stereo_rectification_parameters()


    # undistort left and right image
    undistorted_img_left = cv2.undistort(left_img, Camera_matrix_left, distortion_coef_left, None)
    undistorted_img_right = cv2.undistort(right_img, Camera_matrix_right, distortion_coef_right, None)

    #calling and plotting disparity map
    disparity = disparity_map(undistorted_img_left, undistorted_img_right)
