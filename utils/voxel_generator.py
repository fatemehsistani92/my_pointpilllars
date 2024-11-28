import numpy as np

class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        """
        Args:
            voxel_size (list): Size of each voxel in meters, e.g., [0.16, 0.16, 4.0].
            point_cloud_range (list): Min and max range of the point cloud in 3D space.
            max_num_points (int): Maximum number of points allowed per voxel.
            max_voxels (int): Maximum number of voxels allowed in a frame.
        """
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.grid_size = np.round(
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    def generate(self, points):
        """
        Voxelize the given point cloud.

        Args:
            points (np.ndarray): Array of shape (N, D), where N is the number of points and D is the number of features (e.g., x, y, z, intensity).

        Returns:
            voxel_features (np.ndarray): Features for each voxel (voxels x max_points_per_voxel x feature_dim).
            voxel_coords (np.ndarray): Coordinates of each voxel in the voxel grid.
            voxel_count (int): Number of voxels generated.
        """
        # Filter points outside the point cloud range
        mask = np.all(
            (points[:, :3] >= self.point_cloud_range[:3]) &
            (points[:, :3] <= self.point_cloud_range[3:]),
            axis=1
        )
        points = points[mask]

        # Compute voxel indices for each point
        voxel_indices = ((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        voxel_coords = np.clip(voxel_indices, 0, self.grid_size - 1)

        # Group points into voxels
        voxels = {}
        for coord, point in zip(voxel_coords, points):
            coord_tuple = tuple(coord)
            if coord_tuple not in voxels:
                voxels[coord_tuple] = []
            if len(voxels[coord_tuple]) < self.max_num_points:
                voxels[coord_tuple].append(point)

        # Limit the number of voxels to max_voxels
        voxels = dict(list(voxels.items())[:self.max_voxels])

        # Prepare output arrays
        voxel_features = np.zeros((len(voxels), self.max_num_points, points.shape[1]), dtype=np.float32)
        voxel_coords = np.zeros((len(voxels), 3), dtype=np.int32)
        for i, (coord, points) in enumerate(voxels.items()):
            voxel_features[i, :len(points)] = points
            voxel_coords[i] = coord

        return voxel_features, voxel_coords, len(voxels)
        
        
if __name__ == "__main__":
    # Example point cloud data: (x, y, z, intensity)
    points = np.array([
        [0.5, 0.5, 0.5, 0.8],
        [1.5, -1.5, 0.5, 0.6],
        [2.5, 0.5, -0.5, 0.4],
        [-0.5, -0.5, -0.5, 0.9],
    ])

    voxel_generator = VoxelGenerator(
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-2, -2, -2, 3, 3, 2],
        max_num_points=5,
        max_voxels=10
    )

    voxels, coords, num_voxels = voxel_generator.generate(points)
    print("Voxel Features:", voxels)
    print("Voxel Coordinates:", coords)
    print("Number of Voxels:", num_voxels)
        

