import trimesh
import numpy as np
import matplotlib.pyplot as plt

def compute_sdf_slice(mesh, z_value=None, resolution=200):
    """
    Compute the signed distance field for a slice perpendicular to the z-axis.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Watertight mesh
    z_value : float, optional
        Z coordinate for the slice (default: middle of mesh)
    resolution : int
        Grid resolution for the SDF computation
    
    Returns:
    --------
    X, Y : ndarray
        Meshgrid coordinates
    sdf : ndarray
        Signed distance field values
    """
    # Get bounding box
    bounds = mesh.bounds
    
    # Set z_value to middle if not specified
    if z_value is None:
        z_value = (bounds[0, 2] + bounds[1, 2]) / 2
    
    # Create 2D grid in XY plane
    x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
    y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create 3D points at the specified z-slice
    points = np.column_stack([X.ravel(), Y.ravel(), 
                              np.full(X.size, z_value)])
    
    # Compute signed distance
    # trimesh uses negative inside, positive outside convention
    sdf = -mesh.nearest.signed_distance(points)
    sdf = sdf.reshape(X.shape)
    
    return X, Y, sdf


def visualize_sdf_slice(mesh, z_value=None, resolution=200, 
                        cmap='RdBu_r', show_contours=True):
    """
    Visualize the SDF at a given z-slice.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Watertight mesh
    z_value : float, optional
        Z coordinate for the slice
    resolution : int
        Grid resolution
    cmap : str
        Colormap name
    show_contours : bool
        Whether to show contour lines
    """
    X, Y, sdf = compute_sdf_slice(mesh, z_value, resolution)
    
    if z_value is None:
        z_value = (mesh.bounds[0, 2] + mesh.bounds[1, 2]) / 2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot SDF as heatmap
    im = ax.pcolormesh(X, Y, sdf, cmap=cmap, shading='auto')
    
    # Add zero-level contour (surface boundary)
    if show_contours:
        contours = ax.contour(X, Y, sdf, levels=[0], colors='black', 
                             linewidths=2)
        ax.clabel(contours, inline=True, fontsize=10)
        
        # Additional contour lines
        ax.contour(X, Y, sdf, levels=10, colors='gray', 
                  linewidths=0.5, alpha=0.5)
    
    plt.colorbar(im, ax=ax, label='Signed Distance')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'SDF at z = {z_value:.3f}')
    ax.set_aspect('equal')
    
    return fig, ax


def visualize_multiple_slices(mesh, num_slices=5, resolution=150):
    """
    Visualize SDF at multiple z-slices.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Watertight mesh
    num_slices : int
        Number of slices to visualize
    resolution : int
        Grid resolution per slice
    """
    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
    z_values = np.linspace(z_min, z_max, num_slices)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
    if num_slices == 1:
        axes = [axes]
    
    for i, (ax, z_val) in enumerate(zip(axes, z_values)):
        X, Y, sdf = compute_sdf_slice(mesh, z_val, resolution)
        
        im = ax.pcolormesh(X, Y, sdf, cmap='RdBu_r', shading='auto')
        ax.contour(X, Y, sdf, levels=[0], colors='black', linewidths=2)
        
        ax.set_title(f'z = {z_val:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        plt.colorbar(im, ax=ax, label='SDF')
    
    plt.tight_layout()
    return fig, axes


def compute_sdf_volume(mesh, resolution=100):
    """
    Compute full 3D SDF volume.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Watertight mesh
    resolution : int
        Grid resolution per dimension
    
    Returns:
    --------
    grid_points : ndarray
        3D grid coordinates
    sdf_volume : ndarray
        3D SDF values
    """
    bounds = mesh.bounds
    
    x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
    y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
    z = np.linspace(bounds[0, 2], bounds[1, 2], resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Compute signed distance
    sdf = -mesh.nearest.signed_distance(grid_points)
    sdf_volume = sdf.reshape(X.shape)
    
    return (X, Y, Z), sdf_volume


# Example usage
if __name__ == "__main__":
    # Load your STL file
    mesh1 = trimesh.load("./data-rsc/0-raw/RSC_Cell.stl")
    mesh2 = trimesh.load("./data-rsc/0-raw/RSC_Lid.stl")
    
    mesh = trimesh.util.concatenate([mesh1, mesh2])

    # Ensure mesh is watertight
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight!")
    
    # Visualize single slice
    fig1, ax1 = visualize_sdf_slice(mesh, z_value=None, resolution=200)
    
    # # Visualize multiple slices
    # fig2, axes2 = visualize_multiple_slices(mesh, num_slices=5, resolution=150)
    
    plt.show()