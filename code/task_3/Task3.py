import numpy as np
import cv2
import os.path
import math
from matplotlib import pyplot as plt

def get_images() :
    """This function will read the images from the task_2 folder and return them

    return format: image_one , image_two
    """

    left_img = cv2.imread('../../images/task_3_and_4/left_0.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('../../images/task_3_and_4/right_0.png', cv2.IMREAD_GRAYSCALE)
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


def detect_features(img):
    """This function used ORB class to detect feature points on an Image
        ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance.
        First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them.
        It also use pyramid to produce multiscale-features.

       args:
           img: image on which you need to detect the key feature points

       return:
           kp : keypoint obtained from ORB
           des : descriptors associated with the keypoints.

       descriptors are features vectors


    """
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    return kp, des

    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # plt.imshow(img2),plt.show()


def local_maxima_suppression(keypoints, radius):
    """This function is used for supressing the local maxima i.e. we are tyring to check for points that are maxima in a particular radius

    args:
        keypoints : keypoints of an image
        radius : in this problem are considering six pixel radius

    return :
        maxima : maxima in the six pixel radius

    """

    maximas = []
    _keypoints = keypoints.copy()

    for k in keypoints:
        is_max = True
        for l in _keypoints:
            if math.dist(k.pt, l.pt) <= 6:
                if l.response > k.response:
                    _keypoints.pop(0)
                    is_max = False
                    break
        if is_max:
            maximas.append(k)

    return maximas


def feature_matching(undistorted_img_left, kp_left, des_left, undistorted_img_right, kp_right, des_right,F):
    """This function matches and plots two images and their corresponding points next to each other

    args:
        undistorted_img_left : undistorted image from left camera
        kp_left : key points of left camera
        des_left : descriptor matrix of left camera
        undistorted_img_right : undistorted image from right camera
        kp_right : key points of right camera
        des_right : descriptor matrix of right camera
        F : fundamental matrix obtained while stereo calibration

    return :
        matches: return the list of accepted matches
        kp_new_left : new key points of left matrix corresponding to accepted matches
        kp_new_right : new key points of right matrix corresponding to accepted matches

    """

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des_left, des_right)

    # trying to reduce error

    accepted_matches = []
    kp_new_left = []
    kp_new_right = []

    # Check epipolar constraints
    for m in matches:
        # Convert points to homogenous form
        x_l, x_r = np.array(kp_left[m.queryIdx].pt + (1,)), np.array(kp_right[m.trainIdx].pt + (1,))

        error = np.matmul(np.matmul(x_r.T, F), x_l).item()
        error_limit = 1
        if error > -error_limit and error < error_limit:
            accepted_matches.append(m)
            kp_new_left.append(kp_left[m.queryIdx])
            kp_new_right.append(kp_right[m.trainIdx])


    matches = accepted_matches

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)



    img3 = cv2.drawMatches(undistorted_img_left, kp_left, undistorted_img_right, kp_right, matches, None, flags=2)
    #plt.imshow(img3)
    cv2.imwrite('../../output/task_3/feature_matching_scene_0.png',img3)

    return matches , kp_new_left , kp_new_right

def calculate_triangulation_points(P_l, P_r, kp_new_left , kp_new_right):
    Pt_l = np.array([[p.pt] for p in kp_new_left])
    Pt_r = np.array([[p.pt] for p in kp_new_right])

    points = cv2.triangulatePoints(P_l , P_r , Pt_l , Pt_r)
    cartesian_pts = points[:3]/points[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(*cartesian_pts)
    plt.show()



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

    # trying detect_features on left image:
    kp_left, des_left = detect_features(undistorted_img_left)
    kp_right, des_right = detect_features(undistorted_img_right)

    #drawing keypoints on the image
    Detected_image_features = cv2.drawKeypoints(undistorted_img_left, kp_left, None)
    #plt.imshow(Detected_image_features)
    cv2.imwrite('../../output/task_3/Detected_image_features_scene_0.png',Detected_image_features)

    # local maxima supression
    local_maxima_supressed_kp_left = local_maxima_suppression(kp_left, 12)
    local_maxima_supressed_kp_right = local_maxima_suppression(kp_right, 12)


    # featurematching
    matches , kp_new_left , kp_new_right = feature_matching(undistorted_img_left, kp_left, des_left, undistorted_img_right, kp_right, des_right,F)

    #triangulation
    calculate_triangulation_points(P_l, P_r, kp_new_left, kp_new_right)
