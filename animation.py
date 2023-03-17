import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from PIL import Image
from scipy import interpolate

n_frames = 201
frame_start = 0
length = 124
heigth = 124
data = np.zeros((n_frames, length + 20, heigth + 20))

fileList = glob.glob("testfiles/Cam2_0_0_*.bmp")
# Read file
for idx, filename in enumerate(fileList):
    data[idx] = np.array(Image.open(filename).convert('L'))

fig = plt.figure()
plt.imshow(data[30], cmap='seismic', animated=True)
plt.colorbar()
plt.xlabel("y")
plt.ylabel("x")
plt.show()