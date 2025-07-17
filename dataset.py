import os
import json
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms as Tf
import numpy as np
# import pykitti # We will likely not use pykitti directly for your new dataset
import open3d as o3d
from utils import transform, se3
from PIL import Image
import re # For parsing filenames

# You might need to add or adjust these utility functions if they don't exactly fit your depth image handling
# from utils.transform import DepthImgGenerator, pcd_projection, binary_projection


class KITTIDepthCompletionDataset(Dataset):
    def __init__(self, basedir: str, batch_size: int, cam_id: int = 2,
                 pcd_sample_num=4096, resize_ratio=(0.5, 0.5), extend_ratio=(2.5, 2.5),
                 pooling_size=5): # Added pooling_size as it's used in KITTI_perturb
        self.basedir = basedir
        self.cam_id = cam_id
        self.resize_ratio = resize_ratio
        self.pcd_sample_num = pcd_sample_num
        self.extend_ratio = extend_ratio
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)


        self.image_dir = os.path.join(basedir, 'image')
        self.velodyne_raw_dir = os.path.join(basedir, 'velodyne_raw')
        self.intrinsics_dir = os.path.join(basedir, 'intrinsics')
        self.groundtruth_depth_dir = os.path.join(basedir, 'groundtruth_depth') # If you want to use GT depth for something

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        # Filter files to only include those for the specified camera ID
        self.image_files = [f for f in self.image_files if f"image_{cam_id:02d}.png" in f]


        # Ensure all corresponding files exist (depth, intrinsics, velodyne_raw)
        self.data_items = []
        for img_file in self.image_files:
            # Extract base filename (e.g., 2011_09_26_drive_0002_sync_image_0000000005)
            # and then replace 'image_XX.png' with appropriate suffixes
            base_name = "_".join(img_file.split('_')[:-2])
            frame_id = img_file.split('_')[-2]
            image_cam_id = int(img_file.split('_')[-1].split('.')[0].replace('image_', ''))


            # Check if this file corresponds to the requested cam_id
            if image_cam_id != self.cam_id:
                continue

            gt_depth_file = f"{base_name}_groundtruth_depth_{frame_id}_image_{self.cam_id:02d}.png"
            velo_depth_file = f"{base_name}_velodyne_raw_{frame_id}_image_{self.cam_id:02d}.png"
            intrinsics_file = f"{base_name}_image_{frame_id}_image_{self.cam_id:02d}.txt" # Intrinsics are often per image

            if os.path.exists(os.path.join(self.groundtruth_depth_dir, gt_depth_file)) and \
               os.path.exists(os.path.join(self.velodyne_raw_dir, velo_depth_file)) and \
               os.path.exists(os.path.join(self.intrinsics_dir, intrinsics_file)):
                self.data_items.append({
                    'image': os.path.join(self.image_dir, img_file),
                    'gt_depth': os.path.join(self.groundtruth_depth_dir, gt_depth_file),
                    'velo_depth': os.path.join(self.velodyne_raw_dir, velo_depth_file),
                    'intrinsics': os.path.join(self.intrinsics_dir, intrinsics_file)
                })
        print(f"Found {len(self.data_items)} valid data items for camera {self.cam_id}.")

        self.resample_tran = Resampler(pcd_sample_num) # Re-use if needed for pseudo-PCD from depth
        self.tensor_tran = ToTensor()
        self.img_tran = Tf.ToTensor()

    def __len__(self):
        return len(self.data_items)

    def _read_intrinsics(self, file_path):
        """Reads intrinsic matrix from the provided .txt file."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Assuming the intrinsic matrix is in the first line, e.g., "P_rect_02: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
        # You'll need to parse this correctly based on the actual format.
        # A common format for K is [fx 0 cx; 0 fy cy; 0 0 1]
        # Example parsing if it's like "K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]" or similar
        K = np.eye(3, dtype=np.float32)
        # This parsing part is crucial and depends on the exact format of your intrinsics files.
        # For KITTI depth completion, it's usually 3x3 or 3x4 projection matrix.
        # Let's assume it's a simple 3x3 matrix in the file, space-separated.
        # Example:
        # lines[0] might contain "7.215377e+02 0.000000e+00 6.095593e+02"
        # lines[1] might contain "0.000000e+00 7.215377e+02 1.728540e+02"
        # lines[2] might contain "0.000000e+00 0.000000e+00 1.000000e+00"

        # A common format for KITTI intrinsics (P_rect_XX) can be extracted from calib_cam_to_cam.txt
        # For your case, if it's in a single line, you might need to extract the relevant values.
        # For simplicity, let's assume the file *only* contains the 3x3 K matrix, space or comma separated.
        # You will need to inspect your actual intrinsics files to implement this correctly.
        try:
            # Assuming a simple 3x3 matrix, one row per line, space separated
            K_values = []
            for line in lines:
                K_values.extend([float(x) for x in line.strip().split()])
            K = np.array(K_values).reshape(3, 3).astype(np.float32)
        except ValueError:
            print(f"Warning: Could not parse intrinsics from {file_path}. Using identity matrix.")
            K = np.eye(3, dtype=np.float32)

        return K

    def _read_depth_png(self, filepath):
        """Reads a depth PNG image and converts it to depth values (meters)."""
        # KITTI depth PNGs are typically 16-bit grayscale, values represent depth * 256.0
        # You need to confirm the scaling factor for your specific PNGs.
        depth_img = Image.open(filepath)
        depth_img = np.array(depth_img, dtype=np.float32) / 256.0 # Assuming 16-bit and factor of 256
        # Set invalid depth values (e.g., 0) to a large number or filter them out later
        depth_img[depth_img == 0] = np.nan # Use NaN for invalid to handle later
        return depth_img # (H, W)

    def _depth_to_pcd(self, depth_img, K):
        """Converts a depth image and intrinsic matrix to a 3D point cloud."""
        # This is a simplified version; for full accuracy, refer to KITTI devkit code.
        rows, cols = depth_img.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points_uvd = np.stack([c, r, depth_img], axis=-1).reshape(-1, 3) # (N, 3)

        # Filter out invalid depth points (NaNs or 0s)
        valid_mask = ~np.isnan(points_uvd[:, 2]) & (points_uvd[:, 2] > 0)
        points_uvd = points_uvd[valid_mask]

        if points_uvd.shape[0] == 0:
            return np.empty((3, 0), dtype=np.float32)

        # Convert to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (points_uvd[:, 0] - cx) * points_uvd[:, 2] / fx
        y = (points_uvd[:, 1] - cy) * points_uvd[:, 2] / fy
        z = points_uvd[:, 2]

        pcd = np.stack([x, y, z], axis=-1) # (N, 3)
        return pcd.T # (3, N)

    def __getitem__(self, index):
        data_info = self.data_items[index]

        raw_img = Image.open(data_info['image']).convert('RGB')
        H, W = raw_img.height, raw_img.width
        RH, RW = round(H * self.resize_ratio[0]), round(W * self.resize_ratio[1])
        raw_img = raw_img.resize([RW, RH], Image.BILINEAR)
        _img = self.img_tran(raw_img) # (3, H, W)

        K_cam = self._read_intrinsics(data_info['intrinsics'])
        # Adjust K_cam for resizing
        K_cam[0, 0] *= self.resize_ratio[1]
        K_cam[1, 1] *= self.resize_ratio[0]
        K_cam[0, 2] *= self.resize_ratio[1]
        K_cam[1, 2] *= self.resize_ratio[0]

        # Read velodyne depth map (sparse depth)
        velo_depth_img_np = self._read_depth_png(data_info['velo_depth'])
        # Convert sparse depth map to a sparse point cloud for uncalibed_pcd
        # Note: This is an approximation. The original code uses actual LiDAR scans.
        # You will need to decide if this approximation is sufficient for your calibration task.
        uncalibed_pcd_from_depth = self._depth_to_pcd(velo_depth_img_np, K_cam) # (3, N)

        # Resample the point cloud
        if uncalibed_pcd_from_depth.shape[1] > 0:
            _uncalibed_pcd = self.resample_tran(uncalibed_pcd_from_depth.T).T # (3, n)
        else:
            _uncalibed_pcd = np.empty((3, 0), dtype=np.float32)


        # Generate uncalibrated depth image from this point cloud (similar to original logic)
        # This is essentially re-projecting the sparse point cloud back into an image
        # This part might need further refinement based on how the model expects 'uncalibed_depth_img'
        _uncalibed_depth_img = torch.zeros(RH, RW, dtype=torch.float32)
        if _uncalibed_pcd.shape[1] > 0:
            _pcd_range = np.linalg.norm(_uncalibed_pcd, axis=0)
            # Ensure pcd_projection can handle the reduced dimensions
            u, v, r, _ = transform.pcd_projection((RH, RW), K_cam, _uncalibed_pcd, _pcd_range)
            # Filter out points outside image bounds
            valid_pixels = (u >= 0) & (u < RW) & (v >= 0) & (v < RH)
            _uncalibed_depth_img[v[valid_pixels], u[valid_pixels]] = torch.from_numpy(r[valid_pixels]).type(torch.float32)

        # Apply pooling to depth images
        # The original KITTI_perturb dataset class applies pooling. We can integrate it here or keep a separate perturb class.
        _uncalibed_depth_img_pooled = self.pooling(_uncalibed_depth_img[None,...])


        _uncalibed_pcd_tensor = self.tensor_tran(_uncalibed_pcd)
        _pcd_range_tensor = self.tensor_tran(np.linalg.norm(_uncalibed_pcd, axis=0)) # Recalculate if needed or pass original


        # For `igt` (ground truth transformation), since you don't have odometry,
        # you'll have to *generate* a perturbation here if you want to test the model's robustness to uncalibrated data.
        # The original `test.py` generates `igt` if `test_perturb_file` doesn't exist.
        # For inference, if you're trying to find the calibration, `igt` is not directly used for the model's *forward* pass,
        # but for evaluating the *output* of the model against a known perturbation.
        # If you are just running inference and don't have a ground truth perturbation to compare against,
        # you can set igt to identity or generate random ones for testing purposes.
        # Let's assume for a test/inference scenario, you might still want a "target" or just run the model.
        # We'll generate a random perturbation for `igt` for this dataset for the purpose of testing the calibnet pipeline.
        # In a real inference scenario where you don't have GT, you wouldn't need `igt`.
        max_deg = 10 # From original args
        max_tran = 0.2 # From original args
        mag_randomly = True # From original args
        transform_gen = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)
        transform_gen.generate_transform() # Generate an internal transformation
        igt = transform_gen.igt.squeeze(0) # (4,4)


        return dict(img=_img,
                    uncalibed_pcd=_uncalibed_pcd_tensor,
                    pcd_range=_pcd_range_tensor, # This might be less relevant if we are not directly using depth_img from GT
                    uncalibed_depth_img=_uncalibed_depth_img_pooled.squeeze(0), # Ensure correct dimensions (H, W)
                    InTran=self.tensor_tran(K_cam),
                    igt=self.tensor_tran(igt) # Ground truth transformation (generated)
                    )

# Re-use these utility classes from your original dataset.py
class Resampler:
    def __init__(self, num):
        self.num = num
    def __call__(self, x: np.ndarray):
        num_points = x.shape[0]
        idx = np.random.permutation(num_points)
        if self.num < 0: return x[idx]
        elif self.num <= num_points:
            idx = idx[:self.num]
            return x[idx]
        else:
            idx = np.hstack([idx,np.random.choice(num_points,self.num-num_points,replace=True)])
            return x[idx]

class ToTensor:
    def __init__(self,type=torch.float):
        self.tensor_type = type
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)

# You can remove KITTIFilter and BaseKITTIDataset from dataset.py if you only use the new one.
# Or keep them but make sure you instantiate KITTIDepthCompletionDataset in test.py.