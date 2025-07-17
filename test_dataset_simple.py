#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from KITTIDepthCompletionDataset import KITTIDepthCompletionDataset

def test_dataset():
    # Your data path
    data_path = '/home/deevia/Desktop/sagar/kitti/depth_kitti/data_depth_selection/depth_selection/val_selection_cropped'
    
    print("Testing dataset loading...")
    print(f"Data path: {data_path}")
    
    # Check if directories exist
    required_dirs = ['image', 'groundtruth_depth', 'velodyne_raw', 'intrinsics']
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.txt')])
            print(f"✓ {dir_name}: {file_count} files")
        else:
            print(f"✗ {dir_name}: Directory not found")
    
    # Test dataset creation
    try:
        dataset = KITTIDepthCompletionDataset(
            basedir=data_path,
            cam_id=2,  # Try camera 2
            pcd_sample_num=1000,
            resize_ratio=(1.0, 1.0),
            extend_ratio=(2.5, 2.5),
            pooling_size=5
        )
        print(f"\n✓ Dataset created successfully with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Dataset is empty! Trying camera 3...")
            dataset = KITTIDepthCompletionDataset(
                basedir=data_path,
                cam_id=3,  # Try camera 3
                pcd_sample_num=1000,
                resize_ratio=(1.0, 1.0),
                extend_ratio=(2.5, 2.5),
                pooling_size=5
            )
            print(f"Camera 3 dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            print("\nTesting first sample...")
            sample = dataset[0]
            
            print("Sample contents:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}, dtype: {value.dtype}")
                    if torch.isnan(value).any():
                        print(f"    WARNING: {key} contains NaN values!")
                    if torch.isinf(value).any():
                        print(f"    WARNING: {key} contains infinite values!")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Check specific important values
            print("\nKey statistics:")
            print(f"  Image range: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
            print(f"  Depth image non-zero pixels: {(sample['depth_img'] > 0).sum()}")
            print(f"  Uncalibrated depth non-zero pixels: {(sample['uncalibed_depth_img'] > 0).sum()}")
            print(f"  Point cloud size: {sample['pcd'].shape}")
            print(f"  IGT determinant: {torch.det(sample['igt'][:3, :3]):.6f}")
            
            return True
        else:
            print("✗ Dataset is empty!")
            return False
            
    except Exception as e:
        print(f"✗ Error creating dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n✓ Dataset test passed!")
    else:
        print("\n✗ Dataset test failed!")
        sys.exit(1)