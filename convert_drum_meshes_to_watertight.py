import trimesh
import numpy as np
import matplotlib.pyplot as plt

def convert_to_watertight(stls_to_combine = ["./data-drum/0-raw/drum_cylinder.stl", "./data-drum/0-raw/drum_walls.stl"], output_stl_path="./data-drum/0-raw/merged_meshes.stl"):

    # Combine the input STL files into a single mesh
    combined = None
    for stl_path in stls_to_combine:
        mesh = trimesh.load(stl_path)
        if combined is None:
            combined = mesh
        else:
            combined = trimesh.util.concatenate([combined, mesh])


    combined.fill_holes()
    combined.merge_vertices()
    combined.export(output_stl_path)

def visualize_sdf_gradients_2d(mesh_path = "./data-drum/0-raw/merged_meshes.stl", resolution = 100):
    """
    Create a 2D slice showing SDF gradient vectors
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Create bounding box slightly larger than mesh
    bounds = mesh.bounds
    padding = 0.1 * (bounds[1] - bounds[0])  # 10% padding
    
    # Create 3D grid
    x = np.linspace(bounds[0][0] - padding[0], 
                    bounds[1][0] + padding[0], resolution)
    y = np.linspace(bounds[0][1] - padding[1], 
                    bounds[1][1] + padding[1], resolution)
    z_slice = (bounds[0][2] + bounds[1][2]) / 2  # Middle slice
    
    # Create 2D grid for visualization
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel(), 
                             np.full(xx.ravel().shape, z_slice)])
    
    # Compute SDF values
    sdf_values = -mesh.nearest.signed_distance(points)
    sdf_grid = sdf_values.reshape(xx.shape)
    
    # Compute gradients using normals
    _,_, face_indices = mesh.nearest.on_surface(points)
    normals = mesh.face_normals[face_indices]
    grad_x_norm = normals[:, 0].reshape(xx.shape)
    grad_y_norm = normals[:, 1].reshape(xx.shape)

    # grad_y, grad_x = np.gradient(sdf_grid)
    
    # # Normalize gradients for visualization
    # magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # grad_x_norm = grad_x / (magnitude + 1e-6)  # Avoid division by zero
    # grad_y_norm = grad_y / (magnitude + 1e-6)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. SDF values
    im1 = axes[0, 0].imshow(sdf_grid, cmap='coolwarm', 
                           extent=[x.min(), x.max(), y.min(), y.max()])
    axes[0, 0].set_title('SDF Values')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Gradient magnitude
    # im2 = axes[0, 1].imshow(magnitude, cmap='viridis',
    #                        extent=[x.min(), x.max(), y.min(), y.max()])
    # axes[0, 1].set_title('Gradient Magnitude')
    # plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Gradient vector field (downsampled for clarity)
    step = resolution // 20  # Show every 20th vector
    X, Y = xx[::step, ::step], yy[::step, ::step]
    U, V = grad_x_norm[::step, ::step], grad_y_norm[::step, ::step]
    
    axes[1, 0].quiver(X, Y, U, V, scale=50)
    axes[1, 0].set_title('Gradient Direction Field')
    axes[1, 0].set_aspect('equal')
    
    # 4. Combined visualization
    axes[1, 1].imshow(sdf_grid, cmap='coolwarm', alpha=0.7,
                     extent=[x.min(), x.max(), y.min(), y.max()])
    axes[1, 1].quiver(X, Y, U, V, color='black', scale=50)
    axes[1, 1].set_title('SDF + Gradients')
    
    plt.savefig('sdf_drum.png', dpi=300)
    
    return sdf_grid, grad_x_norm, grad_y_norm


# Example usage
if __name__ == "__main__":
    # Create Watertight mesh
    convert_to_watertight()

    # Visualize SDF gradients
    visualize_sdf_gradients_2d()