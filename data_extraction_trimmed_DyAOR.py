import numpy as np
import pickle

from scipy import stats

DRUM_ROT_SPEED = 3.1415926535

# --- Take a Z slice of the particle positions and use it to calculate Dynamic Angle of Repose (DyAOR) as a global feature for each timestep
# Take a Z slice of the particle positions and use it to calculate Dynamic Angle of Repose (DyAOR) as a global feature for each timestep
def calculate_DyAOR(positions, wall_node_features, nbins=1000):

    # Get the Z slice of particle positions (e.g., particles within a certain Z range around the drum's mid-plane)
    z_min = wall_node_features[2] - 0.1 # Center Z minus particle 3*diameter as margin
    z_max = wall_node_features[2] + 0.1 # Center Z plus particle 3*diameter as margin
    slice_mask = (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
    positions_slice = positions[slice_mask]

    if len(positions_slice) < 10:
        raise ValueError("Too few particles in the selected slice.")

    x = positions_slice[:, 0]
    y = positions_slice[:, 1]

    # Bin the X coordinates
    x_min, x_max = x.min(), x.max()
    bin_edges = np.linspace(x_min, x_max, nbins + 1)
    bin_indices = np.digitize(x, bins=bin_edges) - 1  # 0-based bins

    surface_x = []
    surface_y = []
    for i in range(nbins):
        in_bin = (bin_indices == i)
        if np.any(in_bin):
            # Find the highest particle in this bin
            y_bin = y[in_bin]
            max_idx = np.argmax(y_bin)
            # Record its actual coordinates (not the bin centre)
            x_bin = x[in_bin]
            surface_x.append(x_bin[max_idx])
            surface_y.append(y_bin[max_idx])

    if len(surface_x) < 3:
        raise ValueError("Not enough surface points for a reliable fit.")

    surface_x = np.array(surface_x)
    surface_y = np.array(surface_y)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(surface_x, surface_y)

    # Angle in degrees (absolute value, orientation depends on rotation direction)
    angle = np.arctan(np.abs(slope)) * 180.0 / np.pi
    return angle

# --- Trimmed extracted data only for ML training ---
# We get rid of everything except particle positions, and we also trim the number of timesteps to 1000

# Load extracted data
with open("./data-drum/1-staging/extracted_data.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Trim the number of timesteps to 1000
extracted_data = extracted_data[:4000]

trimmed_data = []
for snapshot_full in extracted_data:

    snapshot_trimmed = {}
    particle_trimmed = {}
    contacts_pp_trimmed = {}
    contacts_pw_trimmed = {}

    snapshot_trimmed["time"] = snapshot_full["time"]
    snapshot_trimmed["timestep"] = snapshot_full["timestep"]

    # --- Get only needed particle features ---

    particle_full = snapshot_full["particle"]
    particle_trimmed["ids"] = particle_full["ids"]
    particle_trimmed["positions"] = particle_full["positions"]
    particle_trimmed["velocities"] = particle_full["velocities"]
    particle_trimmed["net_forces"] = particle_full["net_forces"] # This will be used as ground truth for training, so we keep it in the trimmed data.
    particle_trimmed["sdf_values"] = particle_full["sdf_values"]
    particle_trimmed["sdf_gradients"] = particle_full["sdf_gradients"]
    particle_trimmed["sdf_distance_vectors"] = particle_full["sdf_distance_vectors"]

    snapshot_trimmed["particle"] = particle_trimmed

    # --- Get only needed contact features ---
    contacts_pp_full = snapshot_full["contacts_particle_particle"]
    contacts_pp_trimmed["contact_ids"] = contacts_pp_full["contact_ids"]
    contacts_pp_trimmed["distance_vector"] = contacts_pp_full["distance_vector"]
    snapshot_trimmed["contacts_particle_particle"] = contacts_pp_trimmed

    contacts_pw_full = snapshot_full["contacts_particle_wall"]
    contacts_pw_trimmed["contact_ids"] = contacts_pw_full["contact_ids"]
    snapshot_trimmed["contacts_particle_wall"] = contacts_pw_trimmed

    snapshot_trimmed["wall_node_features"] = snapshot_full["wall_node_features"]

    snapshot_trimmed["wall_node_features"][2] = 0.125 # Drum CoM Z axis fix
    snapshot_trimmed["wall_node_features"][3] = DRUM_ROT_SPEED * snapshot_trimmed["time"] # rad/s * time = current angle of the drum

    # Our second ground truth for training, DyAOR is a global feature that we want the model to predict
    DyAOR = calculate_DyAOR(snapshot_trimmed["particle"]["positions"], snapshot_trimmed["wall_node_features"])
    snapshot_trimmed["DyAOR"] = np.array([DyAOR])
    trimmed_data.append(snapshot_trimmed)

# Save trimmed data in compressed npz format
output_file = "./data-drum/1-staging/extracted_data_trimmed.npz"
np.savez_compressed(output_file, data=trimmed_data)
print(f"Trimmed data saved to {output_file}")