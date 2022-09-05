import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

image = cv2.imread(r"dog.png", cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(src=image)

# To show the dully formed grayscale image
cv2.imshow("Grayscale Dog Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)
print("Total number of pixels in grayscale image: ", image.shape[0] * image.shape[1])

# To show histogram of normal grayscale image.

plt.hist(x=image.ravel(), bins=256, range=[0, 256], color="crimson")
plt.title("Histogram Showing Pixel Intensities And Counts", color="crimson")
plt.ylabel("Number Of Pixels Belonging To The Pixel Intensity", color="crimson")
plt.xlabel("Pixel Intensity", color="crimson")
plt.show()


# To show enhanced grayscale image.
cv2.imshow("Enhanced Contrast Dog Image", image_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To show the enhanced grayscale image histogram.

plt.hist(image_enhanced.ravel(), 256, [0, 256], color="blue")
plt.title("Pixel Intensities And Counts In Enhanced Image", color="crimson")
plt.ylabel("Number Of Pixels Belonging To Pixel Intensity", color="crimson")
plt.xlabel("Pixel Intensity", color="crimson")
plt.show()


# # Below is a mpi program to enhanced the contrast of the image and give the histogram of it.
# # As it runs simultaneously this leads to only one output on the ide


# if rank == 1:
#     # To show the dully formed grayscale image
#     cv2.imshow("Grayscale Dog Image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print(image.shape)
#     print(
#         "Total number of pixels in grayscale image: ", image.shape[0] * image.shape[1]
#     )


# elif rank == 2:
#     # To show histogram of normal grayscale image.

#     plt.hist(x=image.ravel(), bins=256, range=[0, 256], color="crimson")
#     plt.title("Histogram Showing Pixel Intensities And Counts", color="crimson")
#     plt.ylabel("Number Of Pixels Belonging To The Pixel Intensity", color="crimson")
#     plt.xlabel("Pixel Intensity", color="crimson")
#     plt.show()


# elif rank == 0:
#     # To show enhanced grayscale image.
#     cv2.imshow("Enhanced Contrast Dog Image", image_enhanced)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# elif rank == 3:
#     # To show the enhanced grayscale image histogram.

#     plt.hist(image_enhanced.ravel(), 256, [0, 256], color="blue")
#     plt.title("Pixel Intensities And Counts In Enhanced Image", color="crimson")
#     plt.ylabel("Number Of Pixels Belonging To Pixel Intensity", color="crimson")
#     plt.xlabel("Pixel Intensity", color="crimson")
#     plt.show()
