import os
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as Tf
import numpy as np
import cv2
from PIL import Image
from utils import transform, se3

class KITTIDepthCompletionDataset(Dataset):
    def __init__(self, basedir: str, batch_size: int = 1, cam_id: int = 2,
                 pcd_sample_num: int = 4096, resize_ratio: tuple = (1.0, 1.0),
                 extend_ratio: tuple = (2.5, 2.5), pooling_size: int = 5,
                 max_deg: float = 10.0, max_tran: float = 0.2, mag_randomly: bool = True):
        """
        Dataset for KITTI depth completion data structure with individual files
        
        Args:
            basedir: Root directory containing image/, groundtruth_depth/, velodyne_raw/, intrinsics/
            batch_size: Batch size (should be 1 for testing)
            cam_id: Camera ID (2 or 3)
            pcd_sample_num: Number of points to sample from point cloud
            resize_ratio: Image resize ratio (height, width)
            extend_ratio: Extension ratio for projection
            pooling_size: Max pooling size for depth images
            max_deg: Maximum rotation perturbation in degrees
            max_tran: Maximum translation perturbation in meters
            mag_randomly: Whether to randomize perturbation magnitude
        """
        self.basedir = basedir
        self.cam_id = cam_id
        self.resize_ratio = resize_ratio
        self.extend_ratio = extend_ratio
        self.pcd_sample_num = pcd_sample_num
        
        # Setup transforms
        self.transform_se3 = transform.UniformTransformSE3(max_deg, max_tran, mag_randomly)
        self.tensor_tran = ToTensor()
        self.img_tran = Tf.ToTensor()
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=(pooling_size-1)//2)
        
        # Get all image files for the specified camera
        self.image_dir = os.path.join(basedir, 'image')
        self.depth_dir = os.path.join(basedir, 'groundtruth_depth')
        self.velodyne_dir = os.path.join(basedir, 'velodyne_raw')
        self.intrinsics_dir = os.path.join(basedir, 'intrinsics')
        
        # Find all files matching the camera ID
        self.file_list = []
        if os.path.exists(self.image_dir):
            all_files = os.listdir(self.image_dir)
            for file in all_files:
                if f'_image_0{cam_id}.png' in file:
                    # Extract base name without the image part
                    base_name = file.replace(f'_image_0{cam_id}.png', '')
                    self.file_list.append(base_name)
        
        print(f"Found {len(self.file_list)} files for camera {cam_id}")
        
        # Debug: Print first few filenames to understand the pattern
        if len(self.file_list) > 0:
            print(f"First few base names: {self.file_list[:3]}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        base_name = self.file_list[index]
        
        # Load image
        img_filename = f"{base_name}_image_0{self.cam_id}.png"
        img_path = os.path.join(self.image_dir, img_filename)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        raw_img = Image.open(img_path).convert('RGB')
        
        # Load intrinsics - use the same naming pattern as image files
        intrinsics_filename = f"{base_name}_image_0{self.cam_id}.txt"
        intrinsics_path = os.path.join(self.intrinsics_dir, intrinsics_filename)
        
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
            
        K_cam = np.loadtxt(intrinsics_path).reshape(3, 3).astype(np.float32)
        
        # Load velodyne data (as sparse depth image)
        velodyne_filename = f"{base_name.replace('_image_', '_velodyne_raw_')}_image_0{self.cam_id}.png"
        velodyne_path = os.path.join(self.velodyne_dir, velodyne_filename)
        
        if not os.path.exists(velodyne_path):
            raise FileNotFoundError(f"Velodyne file not found: {velodyne_path}")
            
        # Read velodyne sparse depth image
        velodyne_depth = cv2.imread(velodyne_path, cv2.IMREAD_ANYDEPTH)
        velodyne_depth = velodyne_depth.astype(np.float32) / 256.0  # Convert to meters
        
        # Process image
        H, W = raw_img.height, raw_img.width
        RH = round(H * self.resize_ratio[0])
        RW = round(W * self.resize_ratio[1])
        
        # Adjust intrinsics for resizing
        K_cam_resized = K_cam.copy()
        K_cam_resized[0, :] *= self.resize_ratio[1]  # width scaling
        K_cam_resized[1, :] *= self.resize_ratio[0]  # height scaling
        
        # Resize image and depth
        raw_img = raw_img.resize([RW, RH], Image.BILINEAR)
        velodyne_depth = cv2.resize(velodyne_depth, (RW, RH), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        _img = self.img_tran(raw_img)
        _depth_img = torch.from_numpy(velodyne_depth).float()
        K_cam_tensor = self.tensor_tran(K_cam_resized)
        
        # Create point cloud from sparse depth
        valid_mask = _depth_img > 0
        v_coords, u_coords = torch.nonzero(valid_mask, as_tuple=True)
        depths = _depth_img[valid_mask]
        
        # Convert to 3D points
        if len(depths) > 0:
            # Sample points if too many
            if self.pcd_sample_num > 0 and len(depths) > self.pcd_sample_num:
                indices = torch.randperm(len(depths))[:self.pcd_sample_num]
                u_coords = u_coords[indices]
                v_coords = v_coords[indices]
                depths = depths[indices]
            
            # Convert to camera coordinates
            x = (u_coords.float() - K_cam_resized[0, 2]) * depths / K_cam_resized[0, 0]
            y = (v_coords.float() - K_cam_resized[1, 2]) * depths / K_cam_resized[1, 1]
            z = depths
            
            _calibed_pcd = torch.stack([x, y, z], dim=0)  # (3, N)
            _pcd_range = torch.norm(_calibed_pcd, dim=0)
        else:
            # Handle case with no valid depth points by providing a single dummy point
            _calibed_pcd = torch.zeros((3, 1), dtype=torch.float32)  # (3, 1) dummy point
            _pcd_range = torch.ones(1, dtype=torch.float32) # Corresponding range for the dummy point
        
        # --- ROBUSTNESS CHECK: Ensure _calibed_pcd is always (3, N) with N >= 1 ---
        # This is crucial if `torch.nonzero` or subsequent filtering results in an empty point cloud ((3,0))
        if _calibed_pcd.shape[1] == 0: 
            _calibed_pcd = torch.zeros((3, 1), dtype=torch.float32) # Replace with a single dummy point
            _pcd_range = torch.ones(1, dtype=torch.float32) # Adjust range accordingly
        
        # Generate perturbation for testing
        igt = self.transform_se3.generate_transform() # igt is (1, 6)
        
        # --- FIX: Ensure igt_matrix is (4, 4) not (1, 4, 4) ---
        # se3.exp(igt) returns (1, 4, 4) because igt is (1, 6). Squeezing dim 0 gives (4,4).
        igt_matrix = se3.exp(igt).squeeze(0)  # (4, 4)
        
        # Convert _calibed_pcd to homogeneous coordinates (4, N)
        ones_row = torch.ones((1, _calibed_pcd.shape[1]), dtype=_calibed_pcd.dtype, device=_calibed_pcd.device)
        _calibed_pcd_hom = torch.cat((_calibed_pcd, ones_row), dim=0) # (4, N)
        
        # Perform transformation using the full 4x4 matrix
        _uncalibed_pcd_hom = igt_matrix @ _calibed_pcd_hom # (4, 4) @ (4, N) -> (4, N)
        
        # Extract 3D points (first 3 rows)
        _uncalibed_pcd = _uncalibed_pcd_hom[:3, :] 
        
        # Normalize if the w component (4th row) is not 1 (should ideally remain 1 for rigid transforms, but for robustness)
        w_component = _uncalibed_pcd_hom[3, :] # This line was causing the IndexError
        epsilon = 1e-6 # Small epsilon for numerical stability
        if not torch.allclose(w_component, torch.ones_like(w_component), atol=epsilon):
            # Prevent division by zero if w_component is very small or zero
            w_component_clamped = torch.where(torch.abs(w_component) < epsilon, torch.full_like(w_component, epsilon), w_component)
            _uncalibed_pcd = _uncalibed_pcd / w_component_clamped.unsqueeze(0) # (3,N) / (1,N)
            
        # Create uncalibrated depth image
        _uncalibed_depth_img = torch.zeros_like(_depth_img)
        # Only proceed if there are valid points in _uncalibed_pcd after transformation
        if _uncalibed_pcd.shape[1] > 0: # Check if N is greater than 0
            proj_pcd = K_cam_tensor @ _uncalibed_pcd  # (3, 3) @ (3, N) -> (3, N)
            # Handle cases where proj_pcd[2, :] might be zero or negative to avoid division by zero and invalid projections
            valid_z_mask = proj_pcd[2, :] > 0.001 # Small epsilon to avoid division by zero
            
            if valid_z_mask.sum() > 0: # Only project if there are points with valid depth
                proj_x = (proj_pcd[0, valid_z_mask] / proj_pcd[2, valid_z_mask]).long()
                proj_y = (proj_pcd[1, valid_z_mask] / proj_pcd[2, valid_z_mask]).long()
                
                # Check bounds
                valid_proj = (proj_x >= 0) & (proj_x < RW) & (proj_y >= 0) & (proj_y < RH)
                if valid_proj.sum() > 0: # Only assign if there are points within image bounds
                    _uncalibed_depth_img[proj_y[valid_proj], proj_x[valid_proj]] = _pcd_range[valid_z_mask][valid_proj]
        
        # Apply pooling
        _depth_img = self.pooling(_depth_img.unsqueeze(0).unsqueeze(0)).squeeze(0) # Input (H,W) -> (1,1,H,W) for pooling -> (1,1,H,W) output -> (1,H,W) after squeeze(0)
        _uncalibed_depth_img = self.pooling(_uncalibed_depth_img.unsqueeze(0).unsqueeze(0)).squeeze(0) # Same for uncalibrated
        
        # Create dummy extrinsic transformation (identity for this dataset)
        T_cam2velo = torch.eye(4)
        
        return {
            'img': _img,
            'pcd': _calibed_pcd,
            'pcd_range': _pcd_range,
            'depth_img': _depth_img,
            'uncalibed_pcd': _uncalibed_pcd,
            'uncalibed_depth_img': _uncalibed_depth_img,
            'InTran': K_cam_tensor,
            'ExTran': T_cam2velo,
            'igt': igt_matrix
        }


class ToTensor:
    def __init__(self, type=torch.float):
        self.tensor_type = type
    
    def __call__(self, x: np.ndarray):
        return torch.from_numpy(x).type(self.tensor_type)


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loading...")
    base_dir_path = '/home/deevia/Desktop/sagar/kitti/depth_kitti/data_depth_selection/depth_selection/val_selection_cropped'
    print(f"Data path: {base_dir_path}")

    # Basic check for directories
    required_dirs = ['image', 'groundtruth_depth', 'velodyne_raw', 'intrinsics']
    for d in required_dirs:
        path = os.path.join(base_dir_path, d)
        if os.path.exists(path):
            print(f"✓ {d}: {len(os.listdir(path))} files")
        else:
            print(f"✗ {d}: Not found!")

    try:
        dataset = KITTIDepthCompletionDataset(
            basedir=base_dir_path,
            cam_id=2
        )
        print(f"✓ Dataset created successfully with {len(dataset)} samples")   

        print("\nTesting first sample...")
        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample loaded successfully.")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {value}")
            print("✓ Dataset test passed!")
        else:
            print("✗ Dataset is empty, cannot test sample.")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("✗ Dataset test failed due to missing files!")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        print("✗ Dataset test failed!")