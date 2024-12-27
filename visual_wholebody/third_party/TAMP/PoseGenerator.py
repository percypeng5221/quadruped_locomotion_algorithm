import torch
import numpy as np
import math

class PoseGenerator:
    def __init__(self, num_envs, table_pose_x=0.0, table_pose_y=0.0, table_dims_z=1.0, box_size=0.2, device='cuda'):
        """
        Initialize the PoseGenerator class with environment, table parameters, and device.

        Parameters:
        - num_envs (int): Number of environments
        - table_pose_x (float): X-coordinate of the table position
        - table_pose_y (float): Y-coordinate of the table position
        - table_dims_z (float): Height of the table
        - box_size (float): Size of the box
        - device (str or torch.device): Device to store tensors ('cpu' or 'cuda')
        """
        self.num_envs = num_envs
        self.table_pose_x = table_pose_x
        self.table_pose_y = table_pose_y
        self.table_dims_z = table_dims_z
        self.box_size = box_size
        self.device = torch.device(device)
        
        # Initialize position and quaternion tensors on the specified device
        self.position = torch.empty((num_envs, 3), device=self.device)
        self.quat = torch.empty((num_envs, 4), device=self.device)

    def generate(self, index=None):
        """
        Generate or update position and quaternion tensors for the environments.
        
        Parameters:
        - index (torch.Tensor of shape (num_envs,) or None): Boolean array specifying which indices to regenerate.
          If None, regenerates all values.

        Returns:
        - position (torch.Tensor): Updated position tensor of shape (num_envs, 3)
        - quat (torch.Tensor): Updated quaternion tensor of shape (num_envs, 4)
        """
        if index is None:
            # Generate all positions and quaternions
            self.position[:, 0] = self.table_pose_x + torch.FloatTensor(self.num_envs).uniform_(-0.2, 0.1).to(self.device)
            self.position[:, 1] = self.table_pose_y + torch.FloatTensor(self.num_envs).uniform_(-0.3, 0.3).to(self.device)
            self.position[:, 2] = (self.table_dims_z + 0.5 * self.box_size - 0.06) * torch.ones(self.num_envs, device=self.device)

            # Generate all quaternions
            angles = torch.FloatTensor(self.num_envs).uniform_(-math.pi, math.pi).to(self.device)
            half_sin = torch.sin(angles / 2.0)
            half_cos = torch.cos(angles / 2.0)
            
            axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            self.quat[:, 0] = axis[0] * half_sin  # X component
            self.quat[:, 1] = axis[1] * half_sin  # Y component
            self.quat[:, 2] = axis[2] * half_sin  # Z component
            self.quat[:, 3] = half_cos            # W component
        else:
            # Generate only at indices where index is True
            # Ensure index is on the correct device
            index = index.to(self.device)

            # Generate positions at specified indices
            self.position[index, 0] = self.table_pose_x + torch.FloatTensor(index.sum().item()).uniform_(-0.2, 0.1).to(self.device)
            self.position[index, 1] = self.table_pose_y + torch.FloatTensor(index.sum().item()).uniform_(-0.3, 0.3).to(self.device)
            self.position[index, 2] = self.table_dims_z + 0.5 * self.box_size - 0.07

            # Generate quaternions at specified indices
            angles = torch.FloatTensor(index.sum().item()).uniform_(-math.pi, math.pi).to(self.device)
            half_sin = torch.sin(angles / 2.0)
            half_cos = torch.cos(angles / 2.0)
            
            axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            self.quat[index, 0] = axis[0] * half_sin  # X component
            self.quat[index, 1] = axis[1] * half_sin  # Y component
            self.quat[index, 2] = axis[2] * half_sin  # Z component
            self.quat[index, 3] = half_cos            # W component

        return self.position, self.quat
    
if __name__ == "__main__":

    # Example usage
    num_envs = 10
    pose_generator = PoseGenerator(num_envs, device='cuda')  # specify 'cuda' for GPU, 'cpu' for CPU

    # Generate initial tensors on the specified device
    position, quat = pose_generator.generate()
    print("Initial Position Tensor:\n", position)
    print("\nInitial Quaternion Tensor:\n", quat)

    # Boolean array to specify which indices to update
    index = torch.tensor([False, True, False, True, False, False, True, False, False, True], device=pose_generator.device)

    # Regenerate for the specified boolean indices
    position, quat = pose_generator.generate(index=index)
    print("\nUpdated Position Tensor with selected indices:\n", position)
    print("\nUpdated Quaternion Tensor with selected indices:\n", quat)
