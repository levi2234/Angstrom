import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append("C:/Users/esltjv/Desktop/Pers Projects/pbmm/src")

from angstrom.core.motion_amplifier import MotionAmplifier

amplifier = MotionAmplifier("cpu")
amplifier.load_video(r"C:\Users\levi2\Desktop\Projects\Angstrom\src\angstrom\data\testvideos\baby2.mp4")


# image = get_test_image(512, f1)
image = cv2.imread("C:/Users/levi2/Desktop/Projects/Angstrom/src/angstrom/data/testvideos/image.png")
image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
plt.imshow(image)