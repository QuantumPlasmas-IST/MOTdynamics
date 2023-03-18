import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
fileList = sorted(glob.glob("testfiles/Cam2_0_0_*.bmp"))
n_frames = len(fileList)
length = 144
height = 144
data = np.zeros((n_frames, length , height ))


# Read file
for idx, filename in enumerate(fileList):
    data[idx] = np.array(Image.open(filename).convert('L'))

avg_data = np.mean(data, axis=0)
data_clean = data - avg_data

# Apply Gaussian filter
#data_smooth = np.zeros((n_frames, length , height))
#for idx in range(n_frames):
#    data_smooth[idx] = gaussian_filter(data_clean[idx], sigma=.75)

# Reshape matrices into 1D arrays
n_samples, n_rows, n_cols = data_clean.shape
data_1d = np.reshape(data_clean, (n_samples, n_rows * n_cols))

# Perform SVD-based POD
n_components = 3
svd = TruncatedSVD(n_components)
data_pod = svd.fit_transform(data_1d)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=2)
# Plot the first 4 POD components in each subplot
for i in range(n_components):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    mode = svd.components_[i, :]
    mode = np.reshape(mode, (length, height))
    im = ax.imshow(mode, cmap='seismic')
    ax.set_title(f'Mode {i + 1}')
    fig.colorbar(im, ax=ax)
plt.show()

coeff_smoothed = np.zeros(data_pod.shape)

# Plot the time evolution of the first n_components POD modes
figs, axs = plt.subplots(nrows=n_components, ncols=1)
t = np.arange(n_samples)
for i in range(n_components):
    row = i // 2
    col = i % 2
    ax = axs[i]
    ax.plot(t, data_pod[:, i], '--')
    coeff_smoothed[:, i] = savgol_filter(data_pod[:, i], window_length=10, polyorder=3)
    ax.plot(t, coeff_smoothed[:, i])
    ax.set_title(f'Mode {i + 1}')
plt.show()


np.save('POD_coeff_array.npy', data_pod)
np.save('POD_coeff_array_smooth.npy', coeff_smoothed)

plt.plot(data_pod[:, 0], data_pod[:, 1], '--')
plt.plot(coeff_smoothed[:, 0], coeff_smoothed[:, 1])
plt.xlabel("a1")
plt.ylabel("a2")
plt.show()

plt.plot(data_pod[:, 0], data_pod[:, 2], '--')
plt.plot(coeff_smoothed[:, 0], coeff_smoothed[:, 2])
plt.xlabel("a1")
plt.ylabel("a3")
plt.show()

plt.plot(data_pod[:, 1], data_pod[:, 2], '--')
plt.plot(coeff_smoothed[:, 1], coeff_smoothed[:, 2])
plt.xlabel("a2")
plt.ylabel("a3")
plt.show()

# Create the 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D line
ax.plot(coeff_smoothed[:, 0], coeff_smoothed[:, 1], coeff_smoothed[:, 2])

# Add axis labels and a title
ax.set_xlabel(r'$a_1$')
ax.set_ylabel(r'$a_2$')
ax.set_zlabel(r'$a_3$')
ax.set_title('3D Line Plot')

# Add a legend
ax.legend()

# Show the plot
plt.show()
