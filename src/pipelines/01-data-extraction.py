import pandas as pd
import numpy as np
import pyvista as pv
import trimesh

# Use chunking to handle large CSV files if needed
def read_large_csv_in_chunks(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

def convert_stl_to_csv_connectivity(stl_filename, output_csv_filename):

    # Load the STL file
    mesh = trimesh.load_mesh(stl_filename)

    vertices = mesh.vertices
    faces = mesh.faces  # This is your connectivity matrix!

    # Convert to DataFrame for CSV export if needed
    connectivity_df = pd.DataFrame(faces, columns=['v1', 'v2', 'v3'])
    connectivity_df.to_csv(output_csv_filename, index=False)
    print(f"Converted {stl_filename} to {output_csv_filename} with connectivity information.")

def extract_particle_data(filename, output_npz_filename):
    # Read the CSV file into a DataFrame
    df = pd.concat(read_large_csv_in_chunks(filename))
    
    # Get unique sorted timesteps and particle IDs
    timesteps = sorted(df['Timestep'].unique())
    particles = sorted(df['ParticleID'].unique())
    
    n_timesteps = len(timesteps)
    n_particles = len(particles)
    
    print(f"Detected {n_timesteps} timesteps and {n_particles} particles")
    
    # Create the 3D np array, loaded from dataframe
    xData = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)

    # Create id np array
    idData = np.array(particles, dtype=np.int32)

    # Create type np array (all particles hard-coded to type 1 for now) @TODO: support multiple types
    typeData = np.ones(n_particles, dtype=np.int32)

    # Create radius np array, loaded from dataframe
    radiusData = np.zeros((n_particles), dtype=np.float32)

    # Create density np array (all particles hard-coded to 2500 for now) @TODO: support multiple densities
    densityData = np.full(n_particles, 2500, dtype=np.float32)

    # Create mapping dictionaries
    ts_to_idx = {ts: i for i, ts in enumerate(timesteps)}
    pid_to_idx = {pid: i for i, pid in enumerate(particles)}

    # Populate xData and radiusData
    for _, row in df.iterrows():
        ts_idx = ts_to_idx[row['Timestep']]
        pid_idx = pid_to_idx[row['ParticleID']]
        xData[ts_idx][pid_idx] = [row['X'], row['Y'], row['Z']]

        # Calculate radius from the surface area
        radius = (row['SArea'] / (4 * np.pi)) ** (1/2)
        radiusData[pid_idx] = radius

    # Create the final output dictionary
    particleCoord_Data = {
        "xData": xData,
        "idData": idData,
        "typeData": typeData,
        "radiusData": radiusData,
        "densityData": densityData
    }

    # Save it to file
    np.savez_compressed(output_npz_filename, particleCoord_Data=particleCoord_Data)

    return

def extract_triangles(vertices_files, connectivity_files, output_npz_filename):

    all_triangle_data = []
    triangle_counts = []
    
    # Process each object separately
    for v_file, c_file in zip(vertices_files, connectivity_files):
        # Read data for current object
        vertices_df = pd.read_csv(v_file)
        connectivity_df = pd.read_csv(c_file)
        
        # Get timesteps (assuming same timesteps for all objects)
        timesteps = sorted(vertices_df['Timestep'].unique())
        n_timesteps = len(timesteps)
        n_triangles = len(connectivity_df)
        
        # Initialize array for current object
        obj_triangle_data = np.zeros((n_timesteps, n_triangles, 3, 3))
        connectivity = connectivity_df.values
        
        for t_idx, timestep in enumerate(timesteps):
            # Get vertex coordinates for this timestep
            timestep_vertices = vertices_df[vertices_df['Timestep'] == timestep]
            vertex_coords = timestep_vertices[['X', 'Y', 'Z']].values
            
            # Vectorized assignment
            obj_triangle_data[t_idx, :, 0, :] = vertex_coords[connectivity[:, 0]]
            obj_triangle_data[t_idx, :, 1, :] = vertex_coords[connectivity[:, 1]]
            obj_triangle_data[t_idx, :, 2, :] = vertex_coords[connectivity[:, 2]]
        
        all_triangle_data.append(obj_triangle_data)
        triangle_counts.append(n_triangles)
    
    # Concatenate along the triangle dimension (axis=1)
    triangleCoord_Data = np.concatenate(all_triangle_data, axis=1)
    print(triangleCoord_Data.shape)
    
    np.savez_compressed(output_npz_filename, triangleCoord_Data=triangleCoord_Data)

# Convert particle npz data to vtp files for visualization for every timestep
def convert_particle_npz_to_vtp(npz_filename, output_vtp_prefix):
    # Load your data
    data = np.load(npz_filename, allow_pickle=True)
    particleCoord_Data = data['particleCoord_Data'].item()

    xData = particleCoord_Data["xData"]
    radiusData = particleCoord_Data["radiusData"]

    n_timesteps, n_particles, _ = xData.shape

    # Verify dimensions
    assert radiusData.shape[0] == n_particles, "Radius data must have same number of particles as coordinate data"

    # Create a time series of VTP files
    for timestep in range(n_timesteps):
        # Create point cloud for this timestep
        points = xData[timestep]  # Shape: (n_particles, 3)
        
        # Create PyVista point cloud
        point_cloud = pv.PolyData(points)
        
        # Add particle IDs as scalar data
        particle_ids = np.arange(n_particles)
        point_cloud.point_data["ParticleID"] = particle_ids
        
        # Add radius data as scalar data
        point_cloud.point_data["Radius"] = radiusData
        
        # Add timestep as field data
        point_cloud.field_data["Timestep"] = [timestep]
        
        # Save as VTP
        filename = f"{output_vtp_prefix}_{timestep:04d}.vtp"
        point_cloud.save(filename)
        
        print(f"Saved {filename} with radius data")

    print("Conversion complete!")

# Convert wall npz data to vtp files for visualization
def convert_triangle_npz_to_vtp(npz_filename, output_vtp_prefix):

    data = np.load(npz_filename)
    triangleCoord_Data = data['triangleCoord_Data'] # Shape: (n_timesteps, n_triangles, 3, 3)

    n_timesteps = triangleCoord_Data.shape[0]
    
    print(f"Converting {n_timesteps} timesteps with {triangleCoord_Data.shape[1]} triangles each")
    
    # Process each timestep
    for timestep in range(n_timesteps):
        # Get triangle data for current timestep
        triangle_data_timestep = triangleCoord_Data[timestep]
        
        # Create mesh with deduplication
        n_triangles = triangle_data_timestep.shape[0]
    
        # Reshape to get all vertices for this timestep
        all_vertices = triangle_data_timestep.reshape(-1, 3)  # Shape: (n_triangles * 3, 3)
        
        # Find unique vertices and create mapping
        unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)
        
        # Create faces using the unique vertex indices
        faces = np.empty((n_triangles, 4), dtype=np.int64)
        faces[:, 0] = 3  # Number of points per face
        
        # Map original vertex indices to unique indices
        for i in range(n_triangles):
            start_idx = i * 3
            faces[i, 1:4] = inverse[start_idx:start_idx + 3]
        
        # Create the mesh
        mesh = pv.PolyData(unique_vertices, faces)
        
        # Add timestep as field data
        mesh.field_data["Timestep"] = [timestep]
        
        # Add triangle IDs as cell data
        mesh.cell_data["TriangleID"] = np.arange(n_triangles)
        
        # Save as VTP
        filename = f"{output_vtp_prefix}_{timestep:06d}.vtp"
        mesh.save(filename)
    
    print(f"Converted {n_timesteps} timesteps to VTP files")


# Execute the data extraction pipeline
if __name__ == "__main__":

    # STEP 0: Convert STL files to CSV connectivity files
    # convert_stl_to_csv_connectivity("data/00-geometries/drum_cylinder.stl", "data/00-geometries/drum_cylinder.csv")
    # convert_stl_to_csv_connectivity("data/00-geometries/drum_walls.stl", "data/00-geometries/drum_walls.csv")

    # STEP 1: Extract triangle data from vertices and connectivity CSVs
    # connectivity_files = ['data//00-geometries//drum_cylinder.csv', 'data//00-geometries//drum_walls.csv']
    # vertices_files     = ['data//01-raw//run_0_cylinder.csv', 'data//01-raw//run_0_walls.csv']
    # extract_triangles(vertices_files, connectivity_files, "data/02-preprocessed/wallData_0.npz")

    # STEP 2: Extract particle data from raw CSV
    # extract_particle_data("data//01-raw//run_0_particles.csv", "data/02-preprocessed/particleData_0.npz")

    # STEP OPTIONAL: Convert extracted npz data to vtp files for visualization
    # convert_triangle_npz_to_vtp("data/02-preprocessed/wallData_0.npz", "data/02-preprocessed/wall-visualization/walls_0")
    convert_particle_npz_to_vtp("data/02-preprocessed/particleData_0.npz", "data/02-preprocessed/particle-visualization/particles_0")

