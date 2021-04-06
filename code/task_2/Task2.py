import numpy as np
import cv2
import matplotlib
import glob
import os.path
from matplotlib import pyplot as plt

def get_images():
    """
    This function will read the images from the task_2 folder and return them

    return format: image_one , image_two
    """

    left_img = cv2.imread('../../images/task_2/left_0.png')
    right_img = cv2.imread('../../images/task_2/right_0.png')
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




def get_corners(img):
    """
    This function will take an image of chessboard as input (that we are using to calculating stereo calibration) and return the corners
    of the chess board as output

    args:
        img : image in array format
    return type:
        corner_points : in numpy array format
    """

    return cv2.findChessboardCorners(img, (9,6))[1]

def make_list():
    """ This function will return an array of elements of how we want the point to look in 3D format

    return:

    a numpy array of dimension (9,6,1)

    """

    list = []
    for i in range(6):
        for j in range(9):
            list.append([i, j, 0])

    return np.asarray([list], dtype=np.float32)


def Stereo_calibrate(object_corners, left_corners,right_corners, Camera_matrix_left,distortion_coef_left, Camera_matrix_right, distortion_coef_right):

    """ This function will return the output of the stereo Calibration

    args:
        object_corners : 3d reference of object corners
        left_corners: corners captured by left camera
        right_corners: corners captured by right camera
        Camera_matrix_left: intrinsic matrix of left camera
        Camera_matrix_right: intrinsic matrix of right camera
        distortion_coef_left: distortion coefficient of left camera
        distortion_coef_right: distortion coefficient of right camera

    return:
        ret: return a matrix(we do not need this)
        Camera_matrix_left: intrinsic matrix of left camera
        Camera_matrix_right: intrinsic matrix of right camera
        distortion_coef_left: distortion coefficient of left camera
        distortion_coef_right: distortion coefficient of right camera
        R : rotation matrix of second camera with respect to first camera
        T : translation of second camera with respect to first camera
        E : Essential matrix
        F : Fundamental Matrix
    """
    _, Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right, R, T, E, F = cv2.stereoCalibrate(object_corners,
        left_corners, right_corners, Camera_matrix_left,distortion_coef_left, Camera_matrix_right, distortion_coef_right,
        ((640, 480)), flags=cv2.CALIB_FIX_INTRINSIC)

    return _, Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right, R, T, E, F

def plot_camera(ax, r=np.identity(3), t=np.zeros((3, 1))):
    # Constants assumed
    """This function is used for plotting the camera on the final plot.

    """
    tan_x, tan_y, f = 1, 1, 1

    points = np.asarray([
        (0, 0, 0),
        (-tan_x, -tan_y, 1),
        (0, 0, 0),
        (-tan_x, tan_y, 1),
        (0, 0, 0),
        (tan_x, -tan_y, 1),
        (0, 0, 0),
        (tan_x, tan_y, 1),
        (tan_x, -tan_y, 1),
        (-tan_x, -tan_y, 1),
        (-tan_x, tan_y, 1),
        (tan_x, tan_y, 1)
    ]) * f

    points = np.dot(points, r).T + t

    ax.plot(*points, color='black')

def plot_triangulated_points(triangulate_points):
    """
    This function is used for plotting the triangulated points

    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')

    plot_camera(ax)
    plot_camera(ax, R, T)
    for i in triangulate_points.T:
        x,y,z = i[:3]
        x,y,z = x/i[3],y/i[3],z/i[3]
        ax.scatter(x,y,z)

    plt.show()

def Stereo_rectify(Camera_matrix_left , distortion_coef_left ,  Camera_matrix_right , distortion_coef_right):
    """ This function will return the result of stereo rectify open cv function
    args:
        Camera_matrix_left: intrinsic matrix of left camera
        distortion_coef_left: distortion coefficient of left camera
        Camera_matrix_right: intrinsic matrix of right camera
        distortion_coef_right: distortion coefficient of right camera
    return:
        R_l : rotation left
        R_r : rotation right
        P_l : projection matrix left
        P_r : projection matrix right
        Q : disparity to depth mapping
        Roi_l : cropping something
        Roi_r : cropping something

    """

    R_l,R_r ,P_l , P_r , Q , Roi_l , Roi_r = cv2.stereoRectify(Camera_matrix_left , distortion_coef_left ,  Camera_matrix_right , distortion_coef_right,((640, 480)) ,R ,T)

    return R_l,R_r ,P_l , P_r , Q , Roi_l , Roi_r

def write_arrays(path, arrays):
    """Writes arrays to an XML file using cv2.FileStorage
    Args:
        path: Path to the XML file
        arrays: Dictionary containing arrays indexed by keys
    """
    writer = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for key in arrays:
        writer.write(key, arrays[key])
    writer.release()










if __name__ == "__main__":
    #loading images
    left_img, right_img = get_images()

    #loading camera intrinsic parameters
    Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right = get_camera_intrinsics()

    #get left and right corners
    left_corners = get_corners(left_img)
    right_corners = get_corners(right_img)

    #3d matrix for reference
    object_corners = make_list()

    #stereo_calibration
    _, Camera_matrix_left, distortion_coef_left, Camera_matrix_right, distortion_coef_right, R, T, E, F = Stereo_calibrate(np.array([object_corners]), np.array([left_corners]), np.array([right_corners]), Camera_matrix_left,
    distortion_coef_left, Camera_matrix_right, distortion_coef_right)

    # Storing the stereo calibration result in xml file
    write_arrays(os.path.join('../../parameters/', 'stereo_calibration.xml'),
                 {'Camera_matrix_left': Camera_matrix_left,
                  'distortion_coef_left': distortion_coef_left ,
                  'Camera_matrix_right' : Camera_matrix_right ,
                 'distortion_coef_right' : distortion_coef_right,
                 'R' :R,
                 'T' :T,
                 'E':E,
                 'F': F

                  })

    # calculate Undistorted image points for triangulation
    print(T)

    left_undistort_corners = cv2.undistortPoints(left_corners, Camera_matrix_left, distortion_coef_left)
    right_undistort_corners = cv2.undistortPoints(right_corners, Camera_matrix_right, distortion_coef_right)

    # calculate projection matrix for triangulation

    projection_matrix_left = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1).astype('float32')
    projection_matrix_right = np.concatenate((R, T), axis=1).astype('float32')

    # Calculate Triangulation points

    triangulate_points = cv2.triangulatePoints(projection_matrix_left, projection_matrix_right, left_undistort_corners,
                                               right_undistort_corners)

    # plot the camera and triangulated points

    plot_triangulated_points(triangulate_points)

    # Stereo rectify
    R_l, R_r, P_l, P_r, Q, Roi_l, Roi_r = Stereo_rectify(Camera_matrix_left, distortion_coef_left, Camera_matrix_right,
                                                         distortion_coef_right)

    write_arrays(os.path.join('../../parameters/', 'stereo_rectification.xml'),
                 {'R_l': R_l,
                  'R_r': R_r,
                  'P_l': P_l,
                  'P_r': P_r,
                  'Q': Q,
                  'Roi_l': Roi_l,
                  'Roi_r': Roi_r,
                  })






    img_size = left_img.shape[:2]
    _, crn_l = cv2.findChessboardCorners(left_img, (9,6))
    _, crn_r = cv2.findChessboardCorners(right_img, (9,6))

    undist_crn_l = cv2.undistortPoints(crn_l, Camera_matrix_left, distortion_coef_left)
    undist_crn_r = cv2.undistortPoints(crn_r, Camera_matrix_right, distortion_coef_right)
    projection_l = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1).astype('float32')
    projection_r = np.concatenate((R, T), axis=1).astype('float32')

    # Triangulate points and convert to cartesian points
    points = cv2.triangulatePoints(projection_l, projection_r, undist_crn_l, undist_crn_r)
    cartesian_points = points[:3] / points[3]

    # Create subplot and plot cameras and points
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*cartesian_points)

    # Undistort and rectify images
    left_img_undist = cv2.undistort(left_img, Camera_matrix_left, distortion_coef_left, None)
    right_img_undist = cv2.undistort(right_img, Camera_matrix_right, distortion_coef_right, None)

    maps_l = cv2.initUndistortRectifyMap(Camera_matrix_left, distortion_coef_left, R_l, P_l,
                                         img_size, cv2.CV_32FC1)
    maps_r = cv2.initUndistortRectifyMap(Camera_matrix_right, distortion_coef_right, R_r, P_r,
                                         img_size, cv2.CV_32FC1)
    left_img_rect = cv2.remap(left_img_undist, maps_l[0], maps_l[1], cv2.INTER_LINEAR)
    right_img_rect = cv2.remap(right_img_undist, maps_r[0], maps_r[1], cv2.INTER_LINEAR)

    # Find chessboard corners on each image
    _, crn_undist_l = cv2.findChessboardCorners(left_img_undist, (9,6))
    _, crn_undist_r = cv2.findChessboardCorners(right_img_undist, (9,6))
    _, crn_rect_l = cv2.findChessboardCorners(left_img_rect, (9,6))
    _, crn_rect_r = cv2.findChessboardCorners(right_img_rect, (9,6))

    # Draw chessboard corners

    left_img = cv2.drawChessboardCorners(left_img.copy(), (9,6), crn_l, True)
    right_img = cv2.drawChessboardCorners(right_img.copy(), (9,6), crn_r, True)
    left_img_undist = cv2.drawChessboardCorners(left_img_undist.copy(), (9,6), crn_undist_l, True)
    right_img_undist = cv2.drawChessboardCorners(right_img_undist.copy(), (9,6), crn_undist_r, True)
    left_img_rect = cv2.drawChessboardCorners(left_img_rect.copy(), (9,6), crn_rect_l, True)
    right_img_rect = cv2.drawChessboardCorners(right_img_rect.copy(), (9,6), crn_rect_r, True)

    # Write files
    cv2.imwrite(os.path.join('../../output/task_2/', 'left_img.png'), left_img)
    cv2.imwrite(os.path.join('../../output/task_2/', 'right_img.png'), right_img)
    cv2.imwrite(os.path.join('../../output/task_2/', 'left_img_undist.png'), left_img_undist)
    cv2.imwrite(os.path.join('../../output/task_2/', 'right_img_undist.png'), right_img_undist)
    cv2.imwrite(os.path.join('../../output/task_2/', 'left_img_rect.png'), left_img_rect)
    cv2.imwrite(os.path.join('../../output/task_2/', 'right_img_rect.png'), right_img_rect)







