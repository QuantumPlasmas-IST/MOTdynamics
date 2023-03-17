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
from sklearn.decomposition import TruncatedSVD
from matplotlib.animation import FuncAnimation

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

avg_matrix = np.mean(data, axis=0)

plt.imshow(avg_matrix, cmap='seismic', animated=True)
plt.colorbar()
plt.xlabel("y")
plt.ylabel("x")
plt.show()

plt.imshow(data[30] - avg_matrix, cmap='seismic', animated=True)
plt.colorbar()
plt.xlabel("y")
plt.ylabel("x")
plt.show()

# Reshape matrices into 1D arrays
n_samples, n_rows, n_cols = data.shape
data_1d = np.reshape(data, (n_samples, n_rows * n_cols))

# Perform SVD-based POD
n_components = 4
svd = TruncatedSVD(n_components)
data_pod = svd.fit_transform(data_1d)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2)

# Plot the first 4 POD components in each subplot
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    mode = svd.components_[i, :]
    mode = np.reshape(mode, (144, 144))
    im = ax.imshow(mode, cmap='seismic')
    ax.set_title(f'Mode {i + 1}')
    fig.colorbar(im, ax=ax)

plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2)
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    im = ax.imshow(np.reshape(svd.components_[i, :], (144, 144)), cmap='Spectral')
    ax.set_title(f'Mode {i + 1}')
    fig.colorbar(im, ax=ax)
plt.show()


# Define the update function for the animation
def update(frame):
    im.set_data(np.reshape(svd.components_[1, :], (144, 144))*data_pod[frame, 1])
    ax.set_title(f'frame {frame}')
    return im,


# Create the figure and axis for the animation
fig, ax = plt.subplots()
mode_re = np.reshape(svd.components_[1, :], (144, 144))
im = ax.imshow(mode_re, cmap='seismic')
fig.colorbar(im, ax=ax)

# Create the animation
anim = FuncAnimation(fig, update, interval=80)
plt.show()
