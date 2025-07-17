import argparse
import os
import yaml
import torch
from torch.utils.data.dataloader import DataLoader
from KITTIDepthCompletionDataset import KITTIDepthCompletionDataset
from mylogger import get_logger, print_highlight, print_warning
from CalibNet import CalibNet
import loss as loss_utils
import utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image


def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config", type=str, default='config.yml')
    parser.add_argument("--dataset_path", type=str, default='data/')
    parser.add_argument("--skip_frame", type=int, default=1, help='skip frame of dataset')
    parser.add_argument("--pcd_sample", type=int, default=4096)
    parser.add_argument("--max_deg", type=float, default=10)
    parser.add_argument("--max_tran", type=float, default=0.2)
    parser.add_argument("--mag_randomly", type=bool, default=True)
    # dataloader
    parser.add_argument("--batch_size", type=int, default=1, choices=[1])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    # schedule
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--pretrained", type=str, default='./checkpoint/cam2_oneiter_best.pth')
    parser.add_argument("--log_dir", default='log/')
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/")
    parser.add_argument("--res_dir", type=str, default='./results')
    parser.add_argument("--name", type=str, default='cam2_oneiter')
    # setting
    parser.add_argument("--inner_iter", type=int, default=1)
    parser.add_argument("--save_features", type=bool, default=True, help='Save intermediate features')
    parser.add_argument("--save_vectors", type=bool, default=True, help='Save transformation vectors')
    return parser.parse_args()


def save_feature_maps(features, save_path, prefix):
    """Save feature maps from different stages of the network"""
    os.makedirs(save_path, exist_ok=True)
    
    if isinstance(features, torch.Tensor):
        # Convert to numpy and handle different tensor shapes
        feat_np = features.detach().cpu().numpy()
        
        if len(feat_np.shape) == 4:  # (B, C, H, W)
            feat_np = feat_np.squeeze(0)  # Remove batch dimension
            
        if len(feat_np.shape) == 3:  # (C, H, W)
            # Save individual channels
            for i in range(min(feat_np.shape[0], 8)):  # Save first 8 channels
                channel_feat = feat_np[i]
                plt.figure(figsize=(10, 6))
                plt.imshow(channel_feat, cmap='viridis')
                plt.colorbar()
                plt.title(f'{prefix} Feature Map - Channel {i}')
                plt.savefig(os.path.join(save_path, f'{prefix}_channel_{i}.png'), dpi=150)
                plt.close()
                
            # Save mean across channels
            mean_feat = np.mean(feat_np, axis=0)
            plt.figure(figsize=(10, 6))
            plt.imshow(mean_feat, cmap='viridis')
            plt.colorbar()
            plt.title(f'{prefix} Mean Feature Map')
            plt.savefig(os.path.join(save_path, f'{prefix}_mean.png'), dpi=150)
            plt.close()
        
        # Save raw features as numpy
        np.save(os.path.join(save_path, f'{prefix}_features.npy'), feat_np)


def extract_intermediate_features(model, rgb_img, depth_img):
    """Extract intermediate features from different parts of the network"""
    features = {}
    
    # Hook functions to capture intermediate outputs
    def hook_rgb_resnet(module, input, output):
        features['rgb_backbone'] = output[-1]  # Get final output
        
    def hook_depth_resnet(module, input, output):
        features['depth_backbone'] = output[-1]  # Get final output
        
    def hook_aggregation_conv1(module, input, output):
        features['agg_conv1'] = output
        
    def hook_aggregation_conv2(module, input, output):
        features['agg_conv2'] = output
        
    def hook_aggregation_conv3(module, input, output):
        features['agg_conv3'] = output
    
    # Register hooks
    h1 = model.rgb_resnet.register_forward_hook(hook_rgb_resnet)
    h2 = model.depth_resnet.register_forward_hook(hook_depth_resnet)
    h3 = model.aggregation.conv1.register_forward_hook(hook_aggregation_conv1)
    h4 = model.aggregation.conv2.register_forward_hook(hook_aggregation_conv2)
    h5 = model.aggregation.conv3.register_forward_hook(hook_aggregation_conv3)
    
    # Forward pass
    with torch.no_grad():
        rot_pred, trans_pred = model(rgb_img, depth_img)
    
    # Remove hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    
    return features, rot_pred, trans_pred


def visualize_lidar_on_image(rgb_img, pcd, intrinsics, filename, title="LiDAR Points Overlay"):
    """Visualize LiDAR points projected onto the RGB image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show RGB image
    ax.imshow(rgb_img)
    
    # Project LiDAR points to image plane
    if len(pcd) > 0:
        # Filter points in front of camera
        valid_mask = pcd[:, 2] > 0.1
        pcd = pcd[valid_mask]
        
        if len(pcd) > 0:
            # Project to image coordinates
            projected = (intrinsics @ pcd.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            
            # Filter points within image bounds
            h, w = rgb_img.shape[:2]
            in_bounds = ((projected[:, 0] >= 0) & (projected[:, 0] < w)) & \
                       ((projected[:, 1] >= 0) & (projected[:, 1] < h))
            projected = projected[in_bounds]
            pcd = pcd[in_bounds]
            
            # Color points by depth (z coordinate)
            if len(pcd) > 0:
                colors = plt.cm.viridis((pcd[:, 2] - pcd[:, 2].min()) / 
                                      (pcd[:, 2].max() - pcd[:, 2].min() + 1e-6))
                
                # Plot points
                ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=1, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlim(0, rgb_img.shape[1])
    ax.set_ylim(rgb_img.shape[0], 0)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


def create_comparison_visualization(rgb_img, uncalib_pcd, calib_pcd, gt_pcd, intrinsics, filename):
    """Create a comprehensive comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Convert to numpy
    rgb_np = rgb_img.detach().cpu().numpy().transpose(1, 2, 0)
    intrinsics_np = intrinsics.detach().cpu().numpy()
    
    # Function to project and plot points
    def plot_points_on_image(ax, pcd_np, title, color_map='viridis'):
        ax.imshow(rgb_np)
        
        if len(pcd_np) > 0:
            # Filter points in front of camera
            valid_mask = pcd_np[:, 2] > 0.1
            pcd_filtered = pcd_np[valid_mask]
            
            if len(pcd_filtered) > 0:
                # Project to image coordinates
                projected = (intrinsics_np @ pcd_filtered.T).T
                projected = projected[:, :2] / projected[:, 2:3]
                
                # Filter points within image bounds
                h, w = rgb_np.shape[:2]
                in_bounds = ((projected[:, 0] >= 0) & (projected[:, 0] < w)) & \
                           ((projected[:, 1] >= 0) & (projected[:, 1] < h))
                projected = projected[in_bounds]
                pcd_filtered = pcd_filtered[in_bounds]
                
                if len(pcd_filtered) > 0:
                    # Color points by depth
                    colors = plt.cm.get_cmap(color_map)((pcd_filtered[:, 2] - pcd_filtered[:, 2].min()) / 
                                                       (pcd_filtered[:, 2].max() - pcd_filtered[:, 2].min() + 1e-6))
                    ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=1, alpha=0.7)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, rgb_np.shape[1])
        ax.set_ylim(rgb_np.shape[0], 0)
    
    # Plot each configuration
    plot_points_on_image(axes[0, 0], uncalib_pcd.detach().cpu().numpy().T, 
                        'Uncalibrated LiDAR', 'Reds')
    plot_points_on_image(axes[0, 1], calib_pcd.detach().cpu().numpy().T, 
                        'Calibrated LiDAR', 'Blues')
    plot_points_on_image(axes[1, 0], gt_pcd.detach().cpu().numpy().T, 
                        'Ground Truth LiDAR', 'Greens')
    
    # Overlay comparison
    axes[1, 1].imshow(rgb_np)
    
    # Plot all three with different colors and markers
    for pcd_data, color, label, marker in [
        (uncalib_pcd.detach().cpu().numpy().T, 'red', 'Uncalibrated', 'o'),
        (calib_pcd.detach().cpu().numpy().T, 'blue', 'Calibrated', 's'),
        (gt_pcd.detach().cpu().numpy().T, 'green', 'Ground Truth', '^')
    ]:
        if len(pcd_data) > 0:
            valid_mask = pcd_data[:, 2] > 0.1
            pcd_filtered = pcd_data[valid_mask]
            
            if len(pcd_filtered) > 0:
                projected = (intrinsics_np @ pcd_filtered.T).T
                projected = projected[:, :2] / projected[:, 2:3]
                
                h, w = rgb_np.shape[:2]
                in_bounds = ((projected[:, 0] >= 0) & (projected[:, 0] < w)) & \
                           ((projected[:, 1] >= 0) & (projected[:, 1] < h))
                projected = projected[in_bounds]
                
                if len(projected) > 0:
                    # Sample points for better visualization
                    if len(projected) > 1000:
                        indices = np.random.choice(len(projected), 1000, replace=False)
                        projected = projected[indices]
                    
                    axes[1, 1].scatter(projected[:, 0], projected[:, 1], 
                                     c=color, s=2, alpha=0.6, label=label, marker=marker)
    
    axes[1, 1].set_title('Comparison Overlay', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, rgb_np.shape[1])
    axes[1, 1].set_ylim(rgb_np.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


def save_transformation_vectors(vectors, save_path, iteration):
    """Save transformation vectors and create visualization"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save raw vectors
    np.save(os.path.join(save_path, f'transformation_vectors_iter_{iteration}.npy'), vectors)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rotation components
    ax1.bar(['Rx', 'Ry', 'Rz'], vectors[:3])
    ax1.set_title(f'Rotation Components (Iteration {iteration})')
    ax1.set_ylabel('Radians')
    ax1.grid(True, alpha=0.3)
    
    # Plot translation components
    ax2.bar(['Tx', 'Ty', 'Tz'], vectors[3:])
    ax2.set_title(f'Translation Components (Iteration {iteration})')
    ax2.set_ylabel('Meters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'transformation_vectors_iter_{iteration}.png'), dpi=150)
    plt.close()


def calculate_ground_truth_pcd(original_pcd, igt_inverse):
    """Calculate ground truth point cloud using inverse transformation"""
    # Convert to homogeneous coordinates
    ones_row = torch.ones((1, original_pcd.shape[1]), dtype=original_pcd.dtype, device=original_pcd.device)
    pcd_hom = torch.cat((original_pcd, ones_row), dim=0)
    
    # Apply inverse transformation to get ground truth
    gt_pcd_hom = igt_inverse @ pcd_hom
    gt_pcd = gt_pcd_hom[:3, :] / gt_pcd_hom[3:4, :]
    
    return gt_pcd


def test(args, chkpt: dict, test_loader):
    model = CalibNet(depth_scale=args.scale)
    print("Model name:", model.__class__.__name__)
    device = torch.device(args.device)
    print("Device:", device)
    
    model.to(device)
    model.load_state_dict(chkpt['model'])
    model.eval()
    
    logger = get_logger(f'{args.name}-Test', 
                       os.path.join(args.log_dir, f'{args.name}_test.log'), 
                       mode='w')
    logger.debug(args)
    
    res_npy = np.zeros([len(test_loader), 6])
    print("Starting test loop...")
    
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        try:
            rgb_img = batch['img'].to(device)
            B = rgb_img.size(0)
            
            # Get data from batch
            pcd_range = batch['pcd_range'].to(device)
            uncalibed_pcd = batch['uncalibed_pcd'].to(device)
            uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
            original_pcd = batch['pcd'].to(device)  # Original calibrated PCD
            InTran = batch['InTran'].to(device)
            igt = batch['igt'].to(device)
            
            # Handle single sample batch
            if InTran.dim() == 3:
                InTran = InTran[0]
            
            img_shape = rgb_img.shape[-2:]
            
            # Create sample directory
            sample_dir = os.path.join(args.res_dir, 'detailed_analysis', f'sample_{i:04d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Initialize depth generator
            depth_generator = utils.transform.DepthImgGenerator(
                img_shape, InTran, pcd_range, CONFIG['dataset']['pooling']
            )
            
            # Calculate ground truth point cloud (inverse transformation of uncalibrated)
            igt_inverse = torch.inverse(igt)
            gt_pcd = calculate_ground_truth_pcd(uncalibed_pcd, igt_inverse)
            
            Tcl = torch.eye(4).repeat(B, 1, 1).to(device)
            
            # Save initial state
            initial_features, _, _ = extract_intermediate_features(model, rgb_img, uncalibed_depth_img)
            if args.save_features:
                feat_dir = os.path.join(sample_dir, 'features', 'initial')
                for feat_name, feat_data in initial_features.items():
                    save_feature_maps(feat_data, feat_dir, feat_name)
            
            # Store transformation vectors for all iterations
            all_vectors = []
            
            # Iterative refinement
            current_uncalibed_pcd = uncalibed_pcd.clone()
            current_uncalibed_depth_img = uncalibed_depth_img.clone()
            
            for iter_idx in range(args.inner_iter):
                print(f"  Iteration {iter_idx + 1}/{args.inner_iter}")
                
                # Extract features for current iteration
                if args.save_features:
                    features, twist_rot, twist_tsl = extract_intermediate_features(
                        model, rgb_img, current_uncalibed_depth_img)
                    feat_dir = os.path.join(sample_dir, 'features', f'iter_{iter_idx}')
                    for feat_name, feat_data in features.items():
                        save_feature_maps(feat_data, feat_dir, feat_name)
                else:
                    twist_rot, twist_tsl = model(rgb_img, current_uncalibed_depth_img)
                
                # Save transformation vectors
                if args.save_vectors:
                    vectors = torch.cat([twist_rot, twist_tsl], dim=1).squeeze(0).detach().cpu().numpy()
                    all_vectors.append(vectors)
                    save_transformation_vectors(vectors, 
                                              os.path.join(sample_dir, 'vectors'), 
                                              iter_idx)
                
                # Calculate transformation matrix
                iter_Tcl = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))
                
                # Update depth image and point cloud
                current_uncalibed_depth_img, current_uncalibed_pcd = depth_generator(
                    iter_Tcl, current_uncalibed_pcd)
                Tcl = Tcl.bmm(iter_Tcl)
            
            # Save final calibrated results
            final_calibrated_pcd = current_uncalibed_pcd
            
            # Create comprehensive visualization
            create_comparison_visualization(
                rgb_img.squeeze(0),
                uncalibed_pcd.squeeze(0),
                final_calibrated_pcd.squeeze(0), 
                gt_pcd.squeeze(0),
                InTran,
                os.path.join(sample_dir, 'comparison_visualization.png')
            )
            
            # Save individual visualizations
            rgb_np = rgb_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            intrinsics_np = InTran.detach().cpu().numpy()
            
            # Save uncalibrated
            visualize_lidar_on_image(
                rgb_np, 
                uncalibed_pcd.squeeze(0).detach().cpu().numpy().T,
                intrinsics_np,
                os.path.join(sample_dir, 'uncalibrated_overlay.png'),
                'Uncalibrated LiDAR Overlay'
            )
            
            # Save calibrated
            visualize_lidar_on_image(
                rgb_np, 
                final_calibrated_pcd.squeeze(0).detach().cpu().numpy().T,
                intrinsics_np,
                os.path.join(sample_dir, 'calibrated_overlay.png'),
                'Calibrated LiDAR Overlay'
            )
            
            # Save ground truth
            visualize_lidar_on_image(
                rgb_np, 
                gt_pcd.squeeze(0).detach().cpu().numpy().T,
                intrinsics_np,
                os.path.join(sample_dir, 'ground_truth_overlay.png'),
                'Ground Truth LiDAR Overlay'
            )
            
            # Save transformation analysis
            if args.save_vectors and all_vectors:
                all_vectors_np = np.array(all_vectors)
                
                # Plot transformation evolution
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                components = ['Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz']
                for idx, comp in enumerate(components):
                    row, col = idx // 3, idx % 3
                    axes[row, col].plot(all_vectors_np[:, idx], 'o-')
                    axes[row, col].set_title(f'{comp} Evolution')
                    axes[row, col].set_xlabel('Iteration')
                    axes[row, col].set_ylabel('Radians' if idx < 3 else 'Meters')
                    axes[row, col].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, 'transformation_evolution.png'), dpi=150)
                plt.close()
                
                # Save summary
                np.save(os.path.join(sample_dir, 'all_transformation_vectors.npy'), all_vectors_np)
            
            # Calculate error metrics
            dg = Tcl.bmm(igt)
            rot_dx, tsl_dx = loss_utils.gt2euler(dg.squeeze(0).cpu().detach().numpy())
            rot_dx = rot_dx.reshape(-1)
            tsl_dx = tsl_dx.reshape(-1)
            
            res_npy[i, :] = np.abs(np.concatenate([rot_dx, tsl_dx]))
            
            # Save detailed metrics
            metrics = {
                'rotation_error_rad': rot_dx,
                'translation_error_m': tsl_dx,
                'rotation_error_deg': np.degrees(rot_dx),
                'mean_error': res_npy[i, :].mean()
            }
            
            np.save(os.path.join(sample_dir, 'error_metrics.npy'), metrics)
            
            logger.info(f'[{i+1:05d}|{len(test_loader):05d}], '
                       f'rot_err(deg): {np.degrees(rot_dx).mean():.4f}, '
                       f'trans_err(m): {tsl_dx.mean():.4f}, '
                       f'mean_err: {res_npy[i,:].mean():.4f}')
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_loader)} samples")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            print(f"Error processing sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            res_npy[i, :] = np.nan
    
    # Save final results
    print("Saving final results...")
    os.makedirs(args.res_dir, exist_ok=True)
    np.save(os.path.join(args.res_dir, f'{args.name}_results.npy'), res_npy)
    
    # Calculate and log final statistics
    valid_mask = ~np.isnan(res_npy).any(axis=1)
    valid_results = res_npy[valid_mask]
    
    if len(valid_results) > 0:
        angle_errors = np.degrees(valid_results[:, :3])
        trans_errors = valid_results[:, 3:]
        
        # Calculate statistics
        stats = {
            'valid_samples': len(valid_results),
            'total_samples': len(test_loader),
            'angle_errors_deg': {
                'mean': angle_errors.mean(axis=0),
                'std': angle_errors.std(axis=0),
                'median': np.median(angle_errors, axis=0),
                'max': angle_errors.max(axis=0),
                'min': angle_errors.min(axis=0)
            },
            'trans_errors_m': {
                'mean': trans_errors.mean(axis=0),
                'std': trans_errors.std(axis=0),
                'median': np.median(trans_errors, axis=0),
                'max': trans_errors.max(axis=0),
                'min': trans_errors.min(axis=0)
            }
        }
        
        # Save statistics
        np.save(os.path.join(args.res_dir, f'{args.name}_statistics.npy'), stats)
        
        # Log results
        logger.info(f"=== FINAL RESULTS ===")
        logger.info(f'Valid samples: {len(valid_results)}/{len(test_loader)}')
        logger.info(f'Angle errors (deg) - Mean: {stats["angle_errors_deg"]["mean"]}')
        logger.info(f'Angle errors (deg) - Std: {stats["angle_errors_deg"]["std"]}')
        logger.info(f'Translation errors (m) - Mean: {stats["trans_errors_m"]["mean"]}')
        logger.info(f'Translation errors (m) - Std: {stats["trans_errors_m"]["std"]}')
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Valid samples: {len(valid_results)}/{len(test_loader)}")
        print(f"Average angle error: {stats['angle_errors_deg']['mean'].mean():.4f} ± {stats['angle_errors_deg']['std'].mean():.4f} degrees")
        print(f"Average translation error: {stats['trans_errors_m']['mean'].mean():.4f} ± {stats['trans_errors_m']['std'].mean():.4f} meters")
        print(f"Results saved to: {args.res_dir}")
        
    else:
        logger.error("No valid results obtained!")
        print("No valid results obtained!")


if __name__ == "__main__":
    args = options()
    
    # Set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, using CPU')
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        CONFIG = yaml.load(f, yaml.SafeLoader)
    
    # Load checkpoint
    if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
        chkpt = torch.load(args.pretrained, map_location=torch.device('cpu'))
        
        # Update config with checkpoint config
        if 'config' in chkpt:
            CONFIG.update(chkpt['config'])
        
        # Update args with checkpoint args
        if 'args' in chkpt:
            update_args = ['resize_ratio', 'name', 'scale']
            for up_arg in update_args:
                if up_arg in chkpt['args']:
                    setattr(args, up_arg, chkpt['args'][up_arg])
                else:
                    print(f"Warning: '{up_arg}' not found in checkpoint args")
                    # Set default values
                    if up_arg == 'resize_ratio':
                        setattr(args, up_arg, [1.0, 1.0])
                    elif up_arg == 'scale':
                        setattr(args, up_arg, 50.0)
    else:
        raise FileNotFoundError(f'Pretrained checkpoint {os.path.abspath(args.pretrained)} not found!')
    
    print_highlight('Args loaded, creating dataset...')
    
    # Set dataset path
    args.dataset_path = '/home/deevia/Desktop/sagar/kitti/depth_kitti/data_depth_selection/depth_selection/val_selection_cropped'
    
    # Create dataset
    try:
        test_dataset = KITTIDepthCompletionDataset(
            basedir=args.dataset_path,
            batch_size=args.batch_size,
            cam_id=CONFIG['dataset']['cam_id'],
            pcd_sample_num=args.pcd_sample,
            resize_ratio=tuple(args.resize_ratio),
            extend_ratio=tuple(CONFIG['dataset']['extend_ratio']),
            pooling_size=CONFIG['dataset']['pooling'],
            max_deg=args.max_deg,
            max_tran=args.max_tran,
            mag_randomly=args.mag_randomly
        )
        
        print(f"Dataset created with {len(test_dataset)} samples")
        
        if len(test_dataset) == 0:
            print("Error: Dataset is empty! Check your data path and camera ID.")
            exit(1)
            
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        exit(1)
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False
    )
    
    print(f"Starting test with feature extraction: {args.save_features}, vector saving: {args.save_vectors}")
    
    # Run test
    test(args, chkpt, test_dataloader)