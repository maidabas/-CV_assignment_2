import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from solution import Solution

COST1 = 0.5
COST2 = 3.0
WIN_SIZE = 3
DISPARITY_RANGE = 20
##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = '313472417'
ID2 = '315026294'


##########################################################


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def load_data(is_your_data=False):
    # Read the data:
    if is_your_data:
        left_image = mpimg.imread('my_image_left.png')
        right_image = mpimg.imread('my_image_right.png')
    else:
        left_image = mpimg.imread('image_left.png')
        right_image = mpimg.imread('image_right.png')
    return left_image, right_image


def main():
    COST1 = 0.5
    COST2 = 3.0
    WIN_SIZE = 3
    DISPARITY_RANGE = 20

    left_image, right_image = load_data()

    solution = Solution()
    # Compute Sum-Square-Diff distance
    tt = tic()
    sadd = solution.sad_distance(left_image.astype(np.float64),
                                 right_image.astype(np.float64),
                                 win_size=WIN_SIZE,
                                 dsp_range=DISPARITY_RANGE)
    print(f"SADD calculation done in {toc(tt):.4f}[seconds]")

    # Construct naive disparity image
    tt = tic()
    label_map = solution.naive_labeling(sadd)
    print(f"Naive labeling done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.subplot(1, 2, 2)
    plt.imshow(label_map)
    plt.colorbar()
    plt.title('Naive Depth (SAD)')
    plt.show()

    # Smooth disparity image - Dynamic Programming
    tt = tic()
    label_smooth_dp = solution.dp_labeling(sadd, COST1, COST2)
    print(f"Dynamic Programming done in {toc(tt):.4f}[seconds]")

    # plot the left image and the estimated depth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_dp)
    plt.colorbar()
    plt.title('Smooth Depth - DP (SAD)')
    plt.show()

    # Smooth disparity image - Semi-Global Mapping
    tt = tic()
    label_smooth_sgm = solution.sgm_labeling(sadd, COST1, COST2)
    print(f"SGM done in {toc(tt):.4f}[seconds]")

    # Plot Semi-Global Mapping result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_sgm)
    plt.colorbar()
    plt.title('Smooth Depth - SGM (SAD)')
    plt.show()

    # Smooth disparity image - gaussian blur
    tt = tic()
    label_smooth_gb = solution.gaussian_labeling(sadd)
    print(f"SGM done in {toc(tt):.4f}[seconds]")
    # Plot gaussian blur result result:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(left_image)
    plt.title('Source Image')
    plt.subplot(1, 2, 2)
    plt.imshow(label_smooth_gb)
    plt.colorbar()
    plt.title('Smooth Depth - Gaussian Blur (SAD)')
    plt.show()


if __name__ == "__main__":
    main()
