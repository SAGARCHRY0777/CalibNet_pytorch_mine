import numpy as np
import matplotlib.pyplot as plt

# Load point cloud
pcd = np.load('results/visualizations/calibrated_pcd_0000.npy')

# Plot point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1)
plt.show()

# Load and show depth image
depth_img = plt.imread('results/visualizations/calibrated_depth_0000.png')
plt.imshow(depth_img, cmap='viridis')
plt.colorbar()
plt.show()