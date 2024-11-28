import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import PointPillars  # Adjust the import based on your repository structure
from utils.voxel_generator import VoxelGenerator

def load_bin_file(file_path):
    data = np.fromfile(file_path, dtype=np.float32)
    # Truncate extra data
    if data.size % 4 != 0:
        print(f"Warning: Data size {data.size} is not divisible by 4. Truncating.")
        data = data[: (data.size // 4) * 4]
    point_cloud = data.reshape(-1, 4)
    return point_cloud


def inspect_bin_file(file_path):
    """
    Inspect the raw data inside a .bin file to verify its structure.
    """
    # Load the raw binary data
    data = np.fromfile(file_path, dtype=np.float32)
    # Print the size of the data and the first 10 entries
    print(f"Data size: {data.size}")
    print(f"First 10 entries: {data[:10]}")
    return data
    
    

def visualize_voxels(voxel_coords):
    """
    Visualize voxel coordinates in 3D space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], c='r', marker='o')
    ax.set_title("Voxel Grid Visualization")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()

def preprocess(point_cloud):
    """
    Preprocess the point cloud data:
    - Voxelization
    - Feature extraction
    """
    # Initialize the voxel generator
    voxel_generator = VoxelGenerator(
        voxel_size=[0.16, 0.16, 4],  # Adjust voxel size as needed
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],  # Adjust range based on your LiDAR specs
        max_num_points=32,
        max_voxels=20000
    )

    # Generate voxels from the point cloud
    voxels, coordinates, num_points = voxel_generator.generate(point_cloud)

    # Visualize voxel grid (Optional)
    visualize_voxels(coordinates)

    # Prepare features and coordinates for the model
    features = torch.tensor(voxels, dtype=torch.float32)
    coordinates = torch.tensor(coordinates, dtype=torch.int32)

    data_dict = {
        'features': features,
        'coordinates': coordinates,
        'num_points': num_points
    }

    return data_dict

def main():
    # Specify the path to your .bin file
    bin_file_path = '/home/hora/ros2_ws/src/point_cloud_saver/bin_files/000000.bin'  # Update the file path

    # Check if the file exists
    if not os.path.exists(bin_file_path):
        print(f"File not found: {bin_file_path}")
        return

    # Inspect the raw .bin file contents
    print("Inspecting the .bin file...")
    raw_data = inspect_bin_file(bin_file_path)

    # Proceed to load the point cloud
    print("Loading the point cloud...")
    point_cloud = load_bin_file(bin_file_path)
    print(f"Loaded point cloud with shape: {point_cloud.shape}")

    # Preprocess the point cloud data
    data_dict = preprocess(point_cloud)

    # Initialize the PointPillars model
    model = PointPillars(nclasses=3)
    model_path = '/home/hora/ros2_ws/src/pointpillars/pretrained/pointpillars.pth'

    # Load the model weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    # Move data to GPU if available
    if torch.cuda.is_available():
        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].cuda()

    # Run inference
    with torch.no_grad():
        outputs = model(data_dict['features'], data_dict['coordinates'], batch_size=1)
        print("Model outputs:", outputs)
        print("Available keys in outputs:", outputs.keys())

        detections = outputs['detection']
        print("Detections:")
        print(detections)

    # Visualize the results
    visualize(point_cloud, detections)


def visualize(point_cloud, detections):
    """
    Visualize the point cloud and detection results using matplotlib.
    """
    # **Plot the point cloud**
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c='gray')

    # **Plot the bounding boxes**
    for det in detections:
        x, y, z, w, l, h, yaw = det[:7]  # Adjust indices based on your output
        box_corners = get_3d_box(x, y, z, w, l, h, yaw)
        plot_3d_box(ax, box_corners)

    plt.show()

def get_3d_box(x, y, z, w, l, h, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    x_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = l / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    corners = np.dot(R, np.vstack((x_corners, y_corners, z_corners)))
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners

def plot_3d_box(ax, corners):
    """
    Plot the edges of the 3D bounding box.
    """
    # Define the edges connecting the corners
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Upper face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Lower face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    for edge in edges:
        x_coords = [corners[0, edge[0]], corners[0, edge[1]]]
        y_coords = [corners[1, edge[0]], corners[1, edge[1]]]
        z_coords = [corners[2, edge[0]], corners[2, edge[1]]]
        ax.plot(x_coords, y_coords, z_coords, c='r')
        
        
if __name__ == '__main__':
     main()
       
        
