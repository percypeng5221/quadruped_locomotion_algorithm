import os
import random
from time import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

# Assuming these custom modules are available in your environment
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process,
    collision_detect,
    detect_2d_grasp,
    detect_6d_grasp_multi,
)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet


class PointCloudHelper:
    def __init__(self, all_points_num):
        # Precalculate x, y map
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)
        # Get intrinsics
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # Calculate x, y
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()
        # For getting downsampled xyz map
        ymap, xmap = np.meshgrid(
            np.arange(self.output_shape[1]), np.arange(self.output_shape[0])
        )
        factor = 1280 / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self, rgbs: torch.Tensor, depths: torch.Tensor, include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len), dtype=torch.float32
        ).cuda()
        # Calculate z
        idxs = []
        masks = depths > 0
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            # Convert point cloud to xyz maps
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            # Remove zero depth
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            # Random sample if points more than required
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # Save idxs for concat fusion
                idxs.append(cur_idxs)

            # Concat rgb and features after translation
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        # Downsample
        downsample_depths = F.interpolate(
            depths[:, None], size=self.output_shape, mode="nearest"
        ).squeeze(1).cuda()
        # Convert xyzs
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


class GraspDetector:
    def __init__(self):
        # Fixed parameters
        self.checkpoint_path = "/home/percy/princeton/visual_wholebody/third_party/HGGD/realsense_checkpoint"
        self.center_num = 48
        self.anchor_num = 7
        self.anchor_k = 6
        self.anchor_w = 50.0
        self.anchor_z = 20.0
        self.grid_size = 8
        self.all_points_num = 25600
        self.group_num = 512
        self.local_k = 10
        self.ratio = 8
        self.input_h = 360
        self.input_w = 640
        self.local_thres = 0.01
        self.heatmap_thres = 0.01
        self.sigma = 10  # Default value
        self.use_depth = 1  # Default value
        self.use_rgb = 1  # Default value
        self.random_seed = 123

        # Set up point cloud helper
        self.pc_helper = PointCloudHelper(all_points_num=self.all_points_num)

        # Set torch and GPU settings
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
        else:
            raise RuntimeError("CUDA not available")

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Initialize the models
        self.anchornet = AnchorGraspNet(
            in_dim=4, ratio=self.ratio, anchor_k=self.anchor_k
        )
        self.localnet = PointMultiGraspNet(info_size=3, k_cls=self.anchor_num ** 2)

        # Move models to GPU
        self.anchornet = self.anchornet.cuda()
        self.localnet = self.localnet.cuda()

        # Load checkpoint
        self.load_checkpoint(self.checkpoint_path)

        # Set models to evaluation mode
        self.anchornet.eval()
        self.localnet.eval()
    
    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if 'anchor' in checkpoint and 'local' in checkpoint:
            self.anchornet.load_state_dict(checkpoint['anchor'])
            self.localnet.load_state_dict(checkpoint['local'])
            self.anchors = {'gamma': checkpoint['gamma'], 'beta': checkpoint['beta']}
            print(f"-> Loaded checkpoint '{checkpoint_path}'")
        else:
            raise KeyError("Checkpoint does not contain 'anchor' and 'local' keys.")

    def inference(self, rgb_array, depth_array, vis_heatmap=False, vis_grasp=True):
        # Process RGB and depth arrays
        ori_rgb = rgb_array / 255.0
        ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
        ori_rgb = ori_rgb.to(device="cuda", dtype=torch.float32)

        ori_depth = np.clip(depth_array, 0, 1000)
        ori_depth = ori_depth.astype(np.float32)
        ori_depth = torch.from_numpy(ori_depth).T[None]
        ori_depth = ori_depth.to(device="cuda", dtype=torch.float32)

        # Get scene points
        view_points, _, _ = self.pc_helper.to_scene_points(
            ori_rgb, ori_depth, include_rgb=True
        )
        # Get xyz maps
        xyzs = self.pc_helper.to_xyz_maps(ori_depth)

        # Pre-process inputs
        rgb = F.interpolate(ori_rgb, (self.input_w, self.input_h))
        depth = F.interpolate(ori_depth[None], (self.input_w, self.input_h))[0]
        depth = depth / 1000.0
        depth = torch.clip((depth - depth.mean()), -1, 1)
        # Generate 2D input
        x = torch.concat([depth[None], rgb], 1)
        x = x.to(device="cuda", dtype=torch.float32)

        with torch.no_grad():
            # 2D prediction
            pred_2d, perpoint_features = self.anchornet(x)

            loc_map, cls_mask, theta_offset, height_offset, width_offset = anchor_output_process(
                *pred_2d, sigma=self.sigma
            )

            # Detect 2D grasp (x, y, theta)
            rect_gg = detect_2d_grasp(
                loc_map,
                cls_mask,
                theta_offset,
                height_offset,
                width_offset,
                ratio=self.ratio,
                anchor_k=self.anchor_k,
                anchor_w=self.anchor_w,
                anchor_z=self.anchor_z,
                mask_thre=self.heatmap_thres,
                center_num=self.center_num,
                grid_size=self.grid_size,
                grasp_nms=self.grid_size,
                reduce="max",
            )

            # Check 2D result
            if rect_gg.size == 0:
                print("No 2D grasp found")
                return None

            # Visualize heatmap
            if vis_heatmap:
                rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
                resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
                resized_rgb = np.array(
                    resized_rgb.resize((self.input_w, self.input_h))
                ) / 255.0
                depth_t = ori_depth.cpu().numpy().squeeze().T
                plt.subplot(221)
                plt.imshow(rgb_t)
                plt.subplot(222)
                plt.imshow(depth_t)
                plt.subplot(223)
                plt.imshow(loc_map.squeeze().T, cmap="jet")
                plt.subplot(224)
                rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0)
                plt.imshow(rect_rgb)
                plt.tight_layout()
                plt.show()

            # Feature fusion
            points_all = feature_fusion(view_points[..., :3], perpoint_features, xyzs)
            rect_ggs = [rect_gg]
            pc_group, valid_local_centers = data_process(
                points_all,
                ori_depth,
                rect_ggs,
                self.center_num,
                self.group_num,
                (self.input_w, self.input_h),
                min_points=32,
                is_training=False,
            )
            rect_gg = rect_ggs[0]
            # Batch size == 1 when valid
            points_all = points_all.squeeze()

            # Get 2D grasp info for training
            grasp_info = np.zeros((0, 3), dtype=np.float32)
            g_thetas = rect_gg.thetas[None]
            g_ws = rect_gg.widths[None]
            g_ds = rect_gg.depths[None]
            cur_info = np.vstack([g_thetas, g_ws, g_ds])
            grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info = torch.from_numpy(grasp_info).to(
                dtype=torch.float32, device="cuda"
            )

            # LocalNet
            _, pred, offset = self.localnet(pc_group, grasp_info)

            # Detect 6D grasp from 2D output and 6D output
            _, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg,
                pred,
                offset,
                valid_local_centers,
                (self.input_w, self.input_h),
                self.anchors,
                k=self.local_k,
            )

            # Collision detection
            pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
            pred_gg, _ = collision_detect(
                points_all, pred_grasp_from_rect, mode="graspnet"
            )

            # Non-maximum suppression
            pred_gg = pred_gg.nms()

            # Visualize grasp
            if vis_grasp:
                print("Predicted grasp number =", len(pred_gg))
                grasp_geo = pred_gg.to_open3d_geometry_list()
                points = view_points[..., :3].cpu().numpy().squeeze()
                colors = view_points[..., 3:6].cpu().numpy().squeeze()
                vispc = o3d.geometry.PointCloud()
                vispc.points = o3d.utility.Vector3dVector(points)
                vispc.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([vispc] + grasp_geo)

            return pred_gg


# Example usage:
if __name__ == "__main__":
    # Initialize the GraspDetector
    detector = GraspDetector()

    # Load your RGB and depth images as numpy arrays
    # For demonstration, we assume you have 'demo_rgb.png' and 'demo_depth.png'
    rgb_image = np.array(Image.open("/home/percy/princeton/visual_wholebody/third_party/HGGD/images/demo_rgb.png"))
    depth_image = np.array(Image.open("/home/percy/princeton/visual_wholebody/third_party/HGGD/images/demo_depth.png"))
    # print("rgb_image, type(rgb_image), rgb_image.shape: ", rgb_image, type(rgb_image), rgb_image.shape)
    # print("depth_image, type(depth_image), depth_image.shape: ", depth_image, type(depth_image), depth_image.shape)
    print("np.max(depth_image): ", np.max(depth_image))
    # Call the inference method
    pred_gg = detector.inference(rgb_image, depth_image, vis_heatmap=True, vis_grasp=True)

    if pred_gg is not None:
        print("Highest score grasp point: \n", pred_gg[0])
    else:
        print("No grasps detected.")
