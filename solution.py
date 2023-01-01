"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        # Zero pad images
        left_image = np.pad(left_image, ((int((win_size-1)/2), int((win_size-1)/2)), (int((win_size-1)/2), int((win_size-1)/2)), (0,0)), 'constant')
        right_image = np.pad(right_image, ((int((win_size-1)/2), int((win_size-1)/2)), (int((win_size-1)/2)+dsp_range, int((win_size-1)/2)+dsp_range), (0,0)), 'constant')

        # compute ssdd tensor 
        for i in range(ssdd_tensor.shape[0]):
            for j in range(ssdd_tensor.shape[1]):
                for d in disparity_values:
                    left_win = left_image[int(i+int((win_size-1)/2)-(win_size-1)/2):int(i+int((win_size-1)/2)+(win_size-1)/2+1),
                    int(j+int((win_size-1)/2)-(win_size-1)/2):int(j+int((win_size-1)/2)+(win_size-1)/2+1),:]
                    right_win = right_image[int(i+int((win_size-1)/2)-(win_size-1)/2):int(i+int((win_size-1)/2)+(win_size-1)/2+1),
                    int(j+int((win_size-1)/2)+dsp_range-(win_size-1)/2+d):int(j+int((win_size-1)/2)+dsp_range+(win_size-1)/2+d+1),:]
                    ssdd_win = np.sum((left_win - right_win)**2)
                    ssdd_tensor[i,j,d] = ssdd_win



        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        """INSERT YOUR CODE HERE"""
        label_no_smooth = np.argmin(ssdd_tensor,axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        """INSERT YOUR CODE HERE"""
        # initialize M - cost matrix
        M = np.zeros_like(l_slice)
        # Insert first column values:
        l_slice[:,0] = c_slice[:,0]
        for col in num_of_cols:
            for d in num_labels:
                if d == 0:
                    cost_1 = l_slice[d,col-1]
                    cost_2 = p1 + l_slice[d+1, col-1]
                    cost_3 = p2 + np.min(l_slice[d+2:, col-1])
                    M[d,col] = min(cost_1,cost_2,cost_3)
                    l_slice[d,col] = c_slice[d,col] + M[d,col] - np.min(l_slice[:,col-1])
                elif d ==1:
                    cost_1 = l_slice[d,col-1]
                    cost_2 = p1 + min(l_slice[d-1, col-1],l_slice[d+1, col-1])
                    cost_3 = p2 + np.min(l_slice[d+2:, col-1])
                    M[d,col] = min(cost_1,cost_2,cost_3)
                    l_slice[d,col] = c_slice[d,col] + M[d,col] - np.min(l_slice[:,col-1])
                else:
                    cost_1 = l_slice[d,col-1]
                    cost_2 = p1 + min(l_slice[d-1, col-1],l_slice[d+1, col-1])
                    cost_3 = p2 + min(np.min(l_slice[d+2:, col-1]), np.min(l_slice[:d-1, col-1]))
                    M[d,col] = min(cost_1,cost_2,cost_3)
                    l_slice[d,col] = c_slice[d,col] + M[d,col] - np.min(l_slice[:,col-1])

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""

        for row in range(ssdd_tensor.shape[0]):
            l_slice = self.dp_grade_slice(ssdd_tensor[row,:,:], p1, p2)
            l[row,:,:] = l_slice

        return self.naive_labeling(l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

