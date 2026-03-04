import numpy as np
import trimesh
import pickle

from create_rocky_deck import RockyDeck

###############################################
# --- 1. Load STL, Scale, and Transform Mesh --- #
###############################################

# mesh_static = trimesh.load("ball_mill_jar.stl")  # Replace with your STL file path.
# mesh_static.vertices = mesh_static.vertices / 1000.0  # Convert mm to meters. This to match EDEM simulation. Adapt to your case.

# orig_bounds = mesh_static.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
# center_x = (orig_bounds[0, 0] + orig_bounds[1, 0]) / 2.0
# center_z = (orig_bounds[0, 2] + orig_bounds[1, 2]) / 2.0
# top_y = orig_bounds[1, 1]

# def transform_coords(coords):
#     desired_y_shift = 0.00192903  # transformations to match STL with DEM simulation. Adapt to your case.
#     coords = np.atleast_2d(np.array(coords))
#     transformed = coords.copy()
#     transformed[:, 0] = coords[:, 0] - center_x
#     transformed[:, 2] = coords[:, 2] - center_z
#     transformed[:, 1] = coords[:, 1] - top_y + desired_y_shift
#     return transformed

# mesh_static.vertices = transform_coords(mesh_static.vertices)

mesh_static = trimesh.load("./data-drum/0-raw/merged_meshes.stl")

###############################################
# --- 2. Define SDF and SDF Normal Functions --- #
###############################################

def SDF_static(points, target_mesh):
    return -trimesh.proximity.signed_distance(target_mesh, points)

def SDF_normal_direct(point, target_mesh):
    closest, distance, face_index = target_mesh.nearest.on_surface(point.reshape(1, 3))
    normal = target_mesh.face_normals[face_index[0]]
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-8:
        return normal
    return normal / norm_val

def SDF_normal_direct_vectorized(points, target_mesh):
    closest, distance, face_indices = target_mesh.nearest.on_surface(points)
    normals = target_mesh.face_normals[face_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normalized_normals = np.where(norms < 1e-8, normals, normals / norms)
    return normalized_normals

# This is SLOW
# def SDF_gradient_direct(point, target_mesh, epsilon=1e-5):
#     """
#     Computes the gradient of the SDF at a given point using central finite differences.
    
#     Args:
#         point (array-like): A point in space (3D coordinates).
#         target_mesh: A trimesh object.
#         epsilon (float): A small perturbation for finite difference computation.
        
#     Returns:
#         grad (np.array): The gradient vector of the SDF at the given point.
#     """
#     point = np.array(point).reshape(1, 3)
#     grad = np.zeros(3)
#     for i in range(3):
#     # Create perturbation along the i-th axis.
#         d = np.zeros(3)
#         d[i] = epsilon
#         # Compute SDF at slightly displaced points.
#         sdf_plus = -trimesh.proximity.signed_distance(target_mesh, (point + d).reshape(1, 3))[0]
#         sdf_minus = -trimesh.proximity.signed_distance(target_mesh, (point - d).reshape(1, 3))[0]
#         grad[i] = (sdf_plus - sdf_minus) / (2 * epsilon)
#     return grad

def SDF_gradient_direct_vectorized(points, target_mesh, epsilon=1e-5):
    """
    Vectorized computation of the SDF gradient for multiple points using central finite differences.
    
    Args:
        points (array-like): An array of points in space (N x 3 coordinates).
        target_mesh: A trimesh object.
        epsilon (float): A small perturbation for finite difference computation.

    Returns:
        grad (np.array): The gradient vector of the SDF at the given point.
    """
    points = np.array(points)
    N = points.shape[0]
    grad = np.zeros((N, 3))
    
    for i in range(3):
        d = np.zeros((N, 3))
        d[:, i] = epsilon
        
        sdf_plus = -trimesh.proximity.signed_distance(target_mesh, points + d)
        sdf_minus = -trimesh.proximity.signed_distance(target_mesh, points - d)
        
        grad[:, i] = (sdf_plus - sdf_minus) / (2 * epsilon)
    
    return grad


###############################################
# --- 3. Define COM Method for Moving the Mesh --- #
###############################################

# def get_moved_mesh_using_CoM(timestep_index, deck):
#     com_edem = np.array(deck.timestep[timestep_index].geometry['First try jar'].getCoM())
#     com_static = mesh_static.center_mass
#     offset = com_edem - com_static
#     moved_mesh = mesh_static.copy()
#     moved_mesh.vertices = moved_mesh.vertices + offset
#     return moved_mesh

# def SDF_global_moved(x, timestep_index, deck):
#     moved_mesh = get_moved_mesh_using_CoM(timestep_index, deck)
#     return SDF_static(x, moved_mesh)

# timestep_size = 0.01, rad_s = 1.0472 old
def get_rotated_mesh(timestep_index, timestep_size = 0.001, rad_s = 4.188790204666667, axis = [0, 0, 1], center=[0.44, 0.44, 0.0]):
    
    rotated_mesh = mesh_static.copy()

    angle_per_timestep = rad_s * timestep_size
    total_angle = angle_per_timestep * timestep_index
    
    if timestep_index > 0 and abs(total_angle) > 1e-10:
        rotation_matrix = trimesh.transformations.rotation_matrix(
            total_angle,
            axis,
            point=center
        )
        rotated_mesh.apply_transform(rotation_matrix)


    return rotated_mesh

# def SDF_global_rotated(x, timestep_index):
#     rotated_mesh = get_rotated_mesh(timestep_index)
#     return SDF_static(x, rotated_mesh)

###############################################
# --- 4. Load EDEM Data and Define Snapshot Range --- #
###############################################

# decks = []
# for fileName in glob.glob("*.dem"):
#     decks.append(fileName)
# if len(decks) == 0:
#     print("No deck file found in the directory")
#     sys.exit()
# deck_file = decks[0]
# deck = Deck(deck_file)
# print("Using deck file:", deck_file)

# start_timestep = 50000   # starting snapshot index
# last_timestep  = 100000   # adjust as needed for your extraction range
# total_timesteps = deck.numTimesteps
# print("Total timesteps in simulation:", total_timesteps)

# custom_properties_dict = deck.creatorData.simulationPropertyData
# custom_properties_name = np.array(deck.creatorData.simulationCustomPropertyNames)
# custom_properties_num = deck.creatorData.numSimulationCustomProperties

# Load deck from pkl file
with open("./data-drum/1-staging/rocky_deck.pkl", "rb") as f:
    deck = pickle.load(f)

start_timestep = 0
last_timestep  = deck.n_timesteps - 1
total_timesteps = deck.n_timesteps

###############################################
# --- 5. Extract Data from Simulation Snapshots --- #
###############################################

def extract_data(deck, start_timestep, last_timestep):
    extracted_data = []
    prev_joint_normal = None
    prev_joint_tangential = None
    for t in range(start_timestep, last_timestep + 1):
        snapshot = {}
        # snapshot["time"] = deck.timestepValues[t]
        snapshot["time"] = t * 0.001  # assuming timestep size of 0.001s
        snapshot["timestep"] = t  # store snapshot index
        
        # --- Particle Data ---
        # particle = {}
        # particle["num_particles"] = deck.timestep[t].particle[0].getNumParticles()
        # particle["ids"] = deck.timestep[t].particle[0].getIds()
        # particle["positions"] = np.array(deck.timestep[t].particle[0].getPositions())
        # particle["velocities"] = np.array(deck.timestep[t].particle[0].getVelocities())
        # particle["angular_velocities"] = np.array(deck.timestep[t].particle[0].getAngularVelocities())
        # particle["mass"] = np.array(deck.timestep[t].particle[0].getMass())
        # particle["net_forces"] = np.array(deck.timestep[t].particle[0].getForce())
        # particle["torques"] = np.array(deck.timestep[t].particle[0].getTorque())
        # particle["radius"] = np.array(deck.timestep[t].particle[0].getSphereRadii())
        # particle["inertia"] = np.array((2/5) * particle["mass"] * (particle["radius"]**2))

        particle = {}
        particle["num_particles"] = deck.n_particles
        particle["ids"] = deck.p_ids[t]
        particle["positions"] = deck.p_coords[t]
        particle["velocities"] = deck.p_v[t]
        particle["angular_velocities"] = deck.p_av[t]
        particle["mass"] = deck.p_m[t]
        particle["radius"] = deck.p_r[t]
        particle["inertia"] = deck.p_i[t]
        particle["net_forces"] = deck.p_net_forces[t]
        particle["torques"] = deck.p_net_torques[t]

        # particle["sdf_values"] = SDF_global_rotated(particle["positions"], t)

        moved_mesh = get_rotated_mesh(t)
        particle["sdf_values"] = SDF_static(particle["positions"], moved_mesh)
        
        # particle["sdf_normals"] = np.array([SDF_normal_direct(pos, moved_mesh) for pos in particle["positions"]])
        particle["sdf_normals"] = SDF_normal_direct_vectorized(particle["positions"], moved_mesh)

        # particle["sdf_gradients"] = np.array([SDF_gradient_direct(pos, moved_mesh) for pos in particle["positions"]]) # SLOW but accurate
        particle["sdf_gradients"] = SDF_gradient_direct_vectorized(particle["positions"], moved_mesh)

        # Ensure sdf_values is a column vector (shape: [N, 1])
        sdf_values = particle["sdf_values"].reshape(-1, 1)  

        # Get the gradients (shape: [N, 3])
        sdf_gradients = particle["sdf_gradients"]

        # Compute the norm for each gradient vector.
        norms = np.linalg.norm(sdf_gradients, axis=1, keepdims=True)

        # Avoid division by zero: if norm is too small, use the original gradient.
        normalized_gradients = np.where(norms < 1e-8, sdf_gradients, sdf_gradients / norms)

        # Compute the distance vectors.
        particle["sdf_distance_vectors"] = sdf_values * normalized_gradients
        
    
        
        
        snapshot["particle"] = particle
        
        # --- Particle-Particle Contacts ---
        contacts_pp = {}
        try:
            # contacts_pp["contact_ids"] = np.array(deck.timestep[t].contact.surfSurf.getIds()) # Available
            # contacts_pp["normal_forces"] = np.array(deck.timestep[t].contact.surfSurf.getNormalForce()) # Available
            # contacts_pp["tangential_forces"] = np.array(deck.timestep[t].contact.surfSurf.getTangentialForce()) # Available
            # contact_vectors = np.array(deck.timestep[t].contact.surfSurf.getContactVector1())  # Shape (N, 3)
            # contacts_pp["relative_distances"] = np.linalg.norm(contact_vectors, axis=1)  # Shape (N,)
            # contacts_pp["distance_vector"] = np.array(deck.timestep[t].contact.surfSurf.getContactVector1()) * 2

            contacts_pp["contact_ids"] = deck.pp_c_ids[t]
            contacts_pp["normal_forces"] = deck.pp_c_fn[t]
            contacts_pp["tangential_forces"] = deck.pp_c_ft[t]
            contacts_pp["relative_distances"] = deck.pp_relative_dist[t]
            contacts_pp["distance_vector"] = deck.pp_dist_vector[t]
        except KeyError:
            contacts_pp["contact_ids"] = np.empty((0, 2))
            contacts_pp["normal_forces"] = np.empty((0, 3))
            contacts_pp["tangential_forces"] = np.empty((0, 3))
            contacts_pp["relative_distances"] = np.empty((0,))
            contacts_pp["distance_vector"] = np.empty((0,3))
        snapshot["contacts_particle_particle"] = contacts_pp
        
        # --- Particle-Wall Contacts ---
        contacts_pw = {}
        try:
            # contacts_pw["normal_forces"] = np.array(deck.timestep[t].contact.surfGeom.getNormalForce()) # Available
            # contacts_pw["tangential_forces"] = np.array(deck.timestep[t].contact.surfGeom.getTangentialForce()) # Available
            # contacts_pw["contact_ids"] = np.array(deck.timestep[t].contact.surfGeom.getIds()[:, 0]) # Available, take only first column(particle ID that makes a contact with the wall)

            contacts_pw["contact_ids"] = deck.pw_c_ids[t][:, 0]
            contacts_pw["normal_forces"] = deck.pw_c_fn[t]
            contacts_pw["tangential_forces"] = deck.pw_c_ft[t]
            
        except KeyError:
            contacts_pw["normal_forces"] = np.empty((0, 3))
            contacts_pw["tangential_forces"] = np.empty((0, 3))
            contacts_pw["contact_ids"] = np.empty((0,))
            
        snapshot["contacts_particle_wall"] = contacts_pw
        
        # --- Wall Data ---
        # moved_bounds = moved_mesh.bounds
        # wall_center = np.array([
        #     (moved_bounds[0, 0] + moved_bounds[1, 0]) / 2.0,
        #     (moved_bounds[0, 1] + moved_bounds[1, 1]) / 2.0,
        #     (moved_bounds[0, 2] + moved_bounds[1, 2]) / 2.0
        # ])
        # wall_rotational_speed = 300.0  # placeholder value
        # wall_com = np.array(deck.timestep[t].geometry['First try jar'].getCoM())
        wall_rotational_speed = 40 # rpm
        wall_com = [0.44, 0.44, 0.0]
        wall_node_features = np.concatenate([wall_com, np.array([wall_rotational_speed])])
        snapshot["wall_node_features"] = wall_node_features
        
        # --- Energy Dissipation ---
        # For normal energy loss:
        try:
            # pw_n_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Wall Normal Energy Loss')[0][0]].getData()[0]
            # pp_n_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Particle Normal Energy Loss')[0][0]].getData()[0]
            pw_n_loss = deck.pw_n_loss[t]
            pp_n_loss = deck.pp_n_loss[t]
            joint_normal = pw_n_loss + pp_n_loss
        except:
            joint_normal = 0.0
        
        # For tangential energy loss:
        try:
            # pw_t_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Wall Tangential Energy Loss')[0][0]].getData()[0]
            # pp_t_loss = deck.timestep[t].customProperties[np.where(custom_properties_name=='Particle-Particle Tangential Energy Loss')[0][0]].getData()[0]
            pw_t_loss = deck.pw_t_loss[t]
            pp_t_loss = deck.pp_t_loss[t]
            joint_tangential = pw_t_loss + pp_t_loss
        except:
            joint_tangential = 0.0
        
        # Compute incremental loss (difference from previous snapshot)
        if prev_joint_normal is None:
            incremental_normal = 0.0
            incremental_tangential = 0.0
        else:
            incremental_normal = joint_normal - prev_joint_normal
            incremental_tangential = joint_tangential - prev_joint_tangential
        
        prev_joint_normal = joint_normal
        prev_joint_tangential = joint_tangential
        
        snapshot["energy_normal_increment"] = np.array([incremental_normal])
        snapshot["energy_tangential_increment"] = np.array([incremental_tangential])
        
        print("Extracted data for timestep", t)
        extracted_data.append(snapshot)
    return extracted_data

extracted_data = extract_data(deck, start_timestep, last_timestep)
print("Extracted data for", len(extracted_data), "snapshots.")

# Save the extracted data to a file.
with open("./data-drum/1-staging/extracted_data.pkl", "wb") as f:
    pickle.dump(extracted_data, f)
print("Saved extracted data to './data-drum/1-staging/extracted_data.pkl'.")

