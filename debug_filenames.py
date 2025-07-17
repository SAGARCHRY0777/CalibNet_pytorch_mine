#!/usr/bin/env python3

import os

def debug_filenames():
    data_path = '/home/deevia/Desktop/sagar/kitti/depth_kitti/data_depth_selection/depth_selection/val_selection_cropped'
    
    directories = {
        'image': os.path.join(data_path, 'image'),
        'intrinsics': os.path.join(data_path, 'intrinsics'),
        'velodyne_raw': os.path.join(data_path, 'velodyne_raw')
    }
    
    for dir_name, dir_path in directories.items():
        print(f"\n=== {dir_name.upper()} Directory ===")
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.txt')]
            files.sort()
            print(f"Total files: {len(files)}")
            
            # Show first 5 files
            print("First 5 files:")
            for i, file in enumerate(files[:5]):
                print(f"  {i+1}: {file}")
            
            # Show camera 2 files specifically
            cam2_files = [f for f in files if '_02' in f]
            print(f"\nCamera 2 files: {len(cam2_files)}")
            if cam2_files:
                print("First 3 camera 2 files:")
                for i, file in enumerate(cam2_files[:3]):
                    print(f"  {i+1}: {file}")
        else:
            print(f"Directory not found: {dir_path}")
    
    # Test filename matching
    print("\n=== FILENAME MATCHING TEST ===")
    image_dir = directories['image']
    intrinsics_dir = directories['intrinsics']
    
    if os.path.exists(image_dir) and os.path.exists(intrinsics_dir):
        image_files = [f for f in os.listdir(image_dir) if '_image_02.png' in f]
        intrinsics_files = [f for f in os.listdir(intrinsics_dir) if f.endswith('.txt')]
        
        print(f"Image files with camera 2: {len(image_files)}")
        print(f"Intrinsics files: {len(intrinsics_files)}")
        
        # Test matching for first image file
        if image_files:
            test_img = image_files[0]
            base_name = test_img.replace('_image_02.png', '')
            print(f"\nTest image: {test_img}")
            print(f"Base name: {base_name}")
            
            # Try different intrinsics patterns
            patterns = [
                f"{base_name}_sync_image_02.txt",
                f"{base_name}_image_02.txt",
                f"{base_name}.txt",
                f"{base_name}_02.txt"
            ]
            
            print("Testing intrinsics patterns:")
            for pattern in patterns:
                exists = pattern in intrinsics_files
                print(f"  {pattern}: {'✓' if exists else '✗'}")
                
            # Check if any intrinsics file contains the base name
            matching_intrinsics = [f for f in intrinsics_files if base_name in f]
            print(f"\nIntrinsics files containing base name: {len(matching_intrinsics)}")
            for f in matching_intrinsics[:3]:
                print(f"  {f}")

if __name__ == "__main__":
    debug_filenames()