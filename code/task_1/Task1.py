import numpy as np
import cv2
import matplotlib
import glob
import os.path 
from matplotlib import pyplot as plt

def get_images(img_directory = "../../images/task_1/"):
    """
    Using Two different Dictionaries to store left and right images
    """
    left_img  = []  
    right_img = []
    
    left_files = glob.glob(img_directory + 'left_*')
    
    for image in left_files:
        left_img.append(cv2.imread(image))
    
    right_files = glob.glob(img_directory + 'right_*')
    
    for image in right_files:
        right_img.append(cv2.imread(image))
    """
    shape of every image n images folder is 480* 640*3
    
    """
    
    return left_img , right_img



def get_corners(images):
    return np.array([cv2.findChessboardCorners(img, (9,6))[1] for img in images])


"""
shape of left_corners each is 11 ,54,1,2 -> 11 for 11 images and each image will have 9*6 corners with 2 coordinates each 

"""
def make_list():
    """
    This is the function that will make a 3d list
    """
    
    list = []
    for i in range(6):
        for j in range(9):
            list.append([i,j,0])
                
    return np.asarray([list]*11 , dtype = np.float32)


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




if __name__ == '__main__':

    left_images, right_images = get_images()

    left_corners = get_corners(left_images)
    right_corners = get_corners(right_images)


    object_corners = make_list()

    #left camera calibration

    _ , Camera_matrix_left,distortion_coef_left,Rotation_vectors_left,Translation_vector_left = cv2.calibrateCamera(object_corners, left_corners ,((640, 480)),None,None)

    #right camera calibration

    _ , Camera_matrix_right,distortion_coef_right,Rotation_vectors_right,Translation_vector_right = cv2.calibrateCamera(object_corners, right_corners ,((640, 480)),None,None)

    # writing the undistored image to output file

    dst_l = cv2.undistort(left_images[2], Camera_matrix_left, distortion_coef_left, None)
    cv2.imwrite('../../output/task_1/left_2.png', left_images[2])
    cv2.imwrite('../../output/task_1/undistorted_left_2.png',dst_l)

    dst_r = cv2.undistort(right_images[2], Camera_matrix_right, distortion_coef_right, None)
    cv2.imwrite('../../output/task_1/right_2.png', right_images[2])
    cv2.imwrite('../../output/task_1/undistorted_right_2.png',dst_r)


    write_arrays(os.path.join('../../parameters/' ,'left_camera_intrinsics.xml'),
                       {'Camera_matrix_left': Camera_matrix_left, 'distortion_coef_left': distortion_coef_left})
    write_arrays(os.path.join('../../parameters/', 'right_camera_intrinsics.xml'),
                       {'Camera_matrix_right': Camera_matrix_right, 'distortion_coef_right': distortion_coef_right})
