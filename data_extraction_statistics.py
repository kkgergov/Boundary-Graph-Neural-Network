import numpy as np
import matplotlib.pyplot as plt
import pickle

# Easy knobs to customize
BASE_FONTSIZE = 21   # axis labels; ticks/legend use ±1 around this
LINEWIDTH     = 3  # default line thickness for all lines
MARKERSIZE    = 7  # default marker size

PARTICLE_DIAMETER = 0.030 # meters
CONTACT_THRESHOLD_PP = PARTICLE_DIAMETER * 1.5 # Particle-particle contact distance threshold
CONTACT_THRESHOLD_PW_SDF = -(PARTICLE_DIAMETER * 1.5) # Particle-wall contact SDF threshold

plt.rcParams.update({
    # Fonts & text
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE + 1,
    "xtick.labelsize": BASE_FONTSIZE - 1,
    "ytick.labelsize": BASE_FONTSIZE - 1,
    "legend.fontsize": BASE_FONTSIZE - 1,

    # Lines & markers
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,

    # Spines, ticks, grid
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.7,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # Save/export defaults
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 120,   # interactive display
    "savefig.dpi": 600,  # file output (PNG)
    "pdf.fonttype": 42,  # embed TrueType fonts in PDF (no Type 3)
    "ps.fonttype": 42,
})



# Load the extracted data containing snapshots
with open("./data-drum/1-staging/extracted_data.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Container for all SDF values at particle-wall contacts
contact_sdf_values = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Get the SDF values computed for all particles in the snapshot.
    # We flatten the array to ensure it's 1D.
    particle_sdf = np.array(snapshot["particle"]["sdf_values"]).flatten()
    
    # Retrieve the contact indices from the particle-wall contact info.
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if we have any contact indices.
    if contact_ids.size > 0:
        contact_ids = contact_ids.astype(int)
        # If the maximum index is out-of-bound (e.g., 24 for an array of size 24),
        # subtract 1 from all indices to convert from 1-indexed to 0-indexed.
        if contact_ids.max() >= particle_sdf.shape[0]:
            contact_ids = contact_ids - 1
        
        # Get the SDF values corresponding to these contact indices.
        contact_sdfs = particle_sdf[contact_ids]
        contact_sdf_values.extend(contact_sdfs.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_values = np.array(contact_sdf_values)

# Compute and display statistics if any contact SDF values were found.
if contact_sdf_values.size > 0:
    mean_val = np.mean(contact_sdf_values)
    median_val = np.median(contact_sdf_values)
    std_val = np.std(contact_sdf_values)
    min_val = np.min(contact_sdf_values)
    max_val = np.max(contact_sdf_values)
    p5 = np.percentile(contact_sdf_values, 5)
    p95 = np.percentile(contact_sdf_values, 95)

    print("Statistics of SDF values for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the contact SDF values
    plt.figure(figsize=(8, 6))
    plt.hist(contact_sdf_values, bins=50, edgecolor='black')
    plt.xlabel("SDF value (m)")
    plt.ylabel("Frequency")
    #plt.title("Histogram of SDF values for particle-wall contacts")
    # Mark the current threshold (-0.005 m) for reference.
    plt.axvline(CONTACT_THRESHOLD_PW_SDF, color='red', linestyle='--', label="Threshold (-0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF values found in the data.")


# Load the extracted data (assumed saved in "extracted_data.pkl")
with open("./data-drum/1-staging/extracted_data.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Container for all relative distances from particle–particle contacts
relative_distances_list = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-particle contacts dictionary
    contacts_pp = snapshot.get("contacts_particle_particle", {})
    # Get the relative distances array (computed as the Euclidean norm of contact vectors)
    rel_distances = contacts_pp.get("relative_distances", np.empty((0,)))
    
    if rel_distances.size > 0:
        # Flatten in case it's not 1D
        rel_distances = np.array(rel_distances).flatten()
        relative_distances_list.extend(rel_distances.tolist())

# Convert the collected relative distances into a numpy array
relative_distances = np.array(relative_distances_list)

# Compute and display statistics if we found any data
if relative_distances.size > 0:
    mean_val = np.mean(relative_distances)
    median_val = np.median(relative_distances)
    std_val = np.std(relative_distances)
    min_val = np.min(relative_distances)
    max_val = np.max(relative_distances)
    p5 = np.percentile(relative_distances, 5)
    p95 = np.percentile(relative_distances, 95)

    print("Statistics of relative distances for particle-particle contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram for visual inspection
    plt.figure(figsize=(6, 4))
    plt.hist(relative_distances, bins=50, edgecolor='black')
    plt.xlabel("Relative Distance (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of relative distances for particle-particle contacts")
    # Mark the threshold of 2*r (with r = 0.005 m, so 2*r = 0.01 m)
    plt.axvline(CONTACT_THRESHOLD_PP, color='red', linestyle='--', label="Threshold (2*r = 0.01 m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/pp_contacts.png", dpi=300, bbox_inches='tight')
else:
    print("No particle-particle contact relative distance data found.")


with open("./data-drum/1-staging/extracted_data.pkl", "rb") as f:
    extracted_data = pickle.load(f)

# Container for all magnitudes of SDF distance vectors at particle-wall contacts
contact_sdf_distance_magnitudes = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-wall contact information
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    # Access the SDF distance vectors (assumed to be a list or array of 3D vectors)
    sdf_distance_vectors = contacts_pw.get("sdf_distance_vectors", None)
    
    if sdf_distance_vectors is not None:
        # Convert to a numpy array if not already
        sdf_distance_vectors = np.array(sdf_distance_vectors)
        
        # Compute the magnitude (Euclidean norm) for each vector.
        # Assumes each row is a vector with 3 components.
        magnitudes = np.linalg.norm(sdf_distance_vectors, axis=1)
        
        # Add these magnitudes to our container list.
        contact_sdf_distance_magnitudes.extend(magnitudes.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_distance_magnitudes = np.array(contact_sdf_distance_magnitudes)

# Compute and display statistics if any contact SDF distance magnitudes were found.
if contact_sdf_distance_magnitudes.size > 0:
    mean_val = np.mean(contact_sdf_distance_magnitudes)
    median_val = np.median(contact_sdf_distance_magnitudes)
    std_val = np.std(contact_sdf_distance_magnitudes)
    min_val = np.min(contact_sdf_distance_magnitudes)
    max_val = np.max(contact_sdf_distance_magnitudes)
    p5 = np.percentile(contact_sdf_distance_magnitudes, 5)
    p95 = np.percentile(contact_sdf_distance_magnitudes, 95)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the magnitudes
    plt.figure(figsize=(6, 4))
    plt.hist(contact_sdf_distance_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    #plt.title("Histogram of SDF distance vector magnitudes for particle-wall contacts")
    # Mark a threshold line if needed (example threshold at 0.0052 m)
    plt.axvline(CONTACT_THRESHOLD_PW_SDF, color='red', linestyle='--', label=f"Threshold ({CONTACT_THRESHOLD_PW_SDF:.4f} m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/pw_contacts.png", dpi=300, bbox_inches='tight')
else:
    print("No particle-wall contact SDF distance vector values found in the data.")


# Container for all magnitudes of SDF distance vectors at particle-wall contacts
contact_sdf_distance_magnitudes = []

# Loop through each snapshot
for snapshot in extracted_data:
    # Retrieve the particle-wall contact information.
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    
    # Retrieve the contact indices from the particle-wall contact info.
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if we have any contact indices.
    if contact_ids.size > 0:
        contact_ids = contact_ids.astype(int)
        
        # Retrieve the SDF distance vectors (assumed stored under the key "sdf_distance_vectors")
        sdf_distance_vectors = np.array(contacts_pw.get("sdf_distance_vectors", []))
        
        # If the maximum index is out-of-bound (e.g., 24 for an array of size 24),
        # subtract 1 from all indices to convert from 1-indexed to 0-indexed.
        if sdf_distance_vectors.shape[0] > 0 and contact_ids.max() >= sdf_distance_vectors.shape[0]:
            contact_ids = contact_ids - 1
        
        # Get the SDF distance vectors corresponding to these contact indices.
        contact_vectors = sdf_distance_vectors[contact_ids]
        
        # Compute the magnitude (Euclidean norm) for each distance vector.
        magnitudes = np.linalg.norm(contact_vectors, axis=1)
        contact_sdf_distance_magnitudes.extend(magnitudes.tolist())

# Convert the collected list to a numpy array for statistical analysis.
contact_sdf_distance_magnitudes = np.array(contact_sdf_distance_magnitudes)

# Compute and display statistics if any contact SDF distance magnitudes were found.
if contact_sdf_distance_magnitudes.size > 0:
    mean_val = np.mean(contact_sdf_distance_magnitudes)
    median_val = np.median(contact_sdf_distance_magnitudes)
    std_val = np.std(contact_sdf_distance_magnitudes)
    min_val = np.min(contact_sdf_distance_magnitudes)
    max_val = np.max(contact_sdf_distance_magnitudes)
    p5 = np.percentile(contact_sdf_distance_magnitudes, 5)
    p95 = np.percentile(contact_sdf_distance_magnitudes, 95)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts:")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"95th Percentile: {p95:.6f} m")

    # Plot a histogram of the contact SDF distance vector magnitudes
    plt.figure(figsize=(6, 4))
    plt.hist(contact_sdf_distance_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of SDF distance vector magnitudes for particle-wall contacts")
    # Mark a reference threshold line if needed (example threshold at 0.0052 m)
    plt.axvline(0.0052, color='red', linestyle='--', label="Threshold (0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF distance vector values found in the data.")


# Container for all magnitudes of SDF distance vectors at particle-wall contacts across snapshots
all_contact_magnitudes = []

# Loop through each snapshot in the extracted data
for snapshot in extracted_data:
    # Get particle IDs and corresponding SDF distance vectors
    particle_ids = np.array(snapshot["particle"]["ids"])
    sdf_distance_vectors = np.array(snapshot["particle"]["sdf_distance_vectors"])
    
    # Get the contact info for particle-wall contacts
    contacts_pw = snapshot.get("contacts_particle_wall", {})
    contact_ids = contacts_pw.get("contact_ids", np.empty((0,)))
    
    # Only proceed if there are any contact indices.
    if contact_ids.size > 0:
        for cid in contact_ids:
            # Ensure the contact id is an integer.
            cid = int(cid)
            # Find the index in particle_ids where the id equals cid.
            idx_array = np.where(particle_ids == cid)[0]
            if idx_array.size > 0:
                idx = idx_array[0]
                # Retrieve the SDF distance vector corresponding to this particle.
                vec = sdf_distance_vectors[idx]
                # Compute its magnitude.
                mag = np.linalg.norm(vec)
                all_contact_magnitudes.append(mag)

# Convert the collected magnitudes to a NumPy array.
all_contact_magnitudes = np.array(all_contact_magnitudes)

# Compute statistics and plot the distribution if any contact magnitudes were found.
if all_contact_magnitudes.size > 0:
    mean_val = np.mean(all_contact_magnitudes)
    median_val = np.median(all_contact_magnitudes)
    std_val = np.std(all_contact_magnitudes)
    min_val = np.min(all_contact_magnitudes)
    max_val = np.max(all_contact_magnitudes)
    p5 = np.percentile(all_contact_magnitudes, 5)
    p90 = np.percentile(all_contact_magnitudes, 90)

    print("Statistics of SDF distance vector magnitudes for particle-wall contacts (all snapshots):")
    print(f"Mean: {mean_val:.6f} m")
    print(f"Median: {median_val:.6f} m")
    print(f"Standard Deviation: {std_val:.6f} m")
    print(f"Min: {min_val:.6f} m")
    print(f"Max: {max_val:.6f} m")
    print(f"5th Percentile: {p5:.6f} m")
    print(f"90th Percentile: {p90:.6f} m")

    # Plot a histogram of the magnitudes.
    plt.figure(figsize=(6, 4))
    plt.hist(all_contact_magnitudes, bins=50, edgecolor='black')
    plt.xlabel("Magnitude of SDF distance vector (m)")
    plt.ylabel("Frequency")
    plt.title("Histogram of SDF distance vector magnitudes\nfor particle-wall contacts (all snapshots)")
    # Optionally, add a reference threshold line.
    plt.axvline(0.0052, color='red', linestyle='--', label="Threshold (0.0052 m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No particle-wall contact SDF distance vector values found in the data.") 

# Lists to collect durations for each collision type
pp_durations = []  # particle-particle durations
pw_durations = []  # particle-wall durations

# Loop over timesteps 5005 to 15000 (adjust indices as needed)
for i in range(5005, 15000):
    timestep = deck.timestep[i]
    
    # Particle-Particle collisions (surfSurf)
    try:
        # Try to extract start and end times for particle-particle collisions
        start_times_pp = timestep.collision.surfSurf.getStartTime()  # returns an array
        end_times_pp   = timestep.collision.surfSurf.getEndTimes()    # returns an array
        durations_pp = end_times_pp - start_times_pp
        pp_durations.extend(durations_pp.tolist())
    except Exception as e:
        # If extraction fails (e.g., no contacts), skip this timestep for PP collisions
        pass

    # Particle-Wall collisions (surfGeom)
    try:
        # Try to extract start and end times for particle-wall collisions
        start_times_pw = timestep.collision.surfGeom.getStartTime()
        end_times_pw   = timestep.collision.surfGeom.getEndTimes()
        durations_pw = end_times_pw - start_times_pw
        pw_durations.extend(durations_pw.tolist())
    except Exception as e:
        # If extraction fails (e.g., no contacts), skip this timestep for PW collisions
        pass

# Convert lists to NumPy arrays for statistics and plotting
pp_durations = np.array(pp_durations)
pw_durations = np.array(pw_durations)

# Compute basic statistics
pp_mean = np.mean(pp_durations) if pp_durations.size > 0 else 0
pp_std  = np.std(pp_durations) if pp_durations.size > 0 else 0
pw_mean = np.mean(pw_durations) if pw_durations.size > 0 else 0
pw_std  = np.std(pw_durations) if pw_durations.size > 0 else 0
pp_median = np.median(pp_durations)
pw_median = np.median(pw_durations)

# Assuming pp_durations and pw_durations are numpy arrays
pp_90 = np.percentile(pp_durations, 90)
pp_95 = np.percentile(pp_durations, 95)
pw_90 = np.percentile(pw_durations, 90)
pw_95 = np.percentile(pw_durations, 95)

print(f"Particle-Particle collision durations:")
print(f"  90th percentile: {pp_90:.6f} s")
print(f"  95th percentile: {pp_95:.6f} s")

print(f"Particle-Wall collision durations:")
print(f"  90th percentile: {pw_90:.6f} s")
print(f"  95th percentile: {pw_95:.6f} s")

print("Particle-Particle Collision Durations:")
print(f"  Mean: {pp_mean:.6f} s, Std: {pp_std:.6f} s")
print("Particle-Wall Collision Durations:")
print(f"  Mean: {pw_mean:.6f} s, Std: {pw_std:.6f} s")
print(f"Particle-Particle median duration: {pp_median:.6f} s")
print(f"Particle-Wall median duration: {pw_median:.6f} s")



# 1) Filter out any zero or negative durations if they exist (optional but recommended)
pp_durations = pp_durations[pp_durations > 0]
pw_durations = pw_durations[pw_durations > 0]

# 2) Build logarithmically spaced bins.
#    We'll set the lower bound as the min of the durations, the upper bound as the max.
#    You can also hard-code them if you want to ignore outliers.
bins_pp = np.logspace(np.log10(pp_durations.min()),
                      np.log10(pp_durations.max()),
                      50)  # 50 bins
bins_pw = np.logspace(np.log10(pw_durations.min()),
                      np.log10(pw_durations.max()),
                      50)

plt.figure(figsize=(12, 5))

# Particle-Particle
plt.subplot(1, 2, 1)
plt.hist(pp_durations, bins=bins_pp, color='blue', alpha=0.7)
plt.xscale('log')          # Use a log scale on the x-axis
plt.xlabel("Duration (s) [log scale]")
plt.ylabel("Frequency")
plt.title("Particle-Particle Collision Durations")

# Particle-Wall
plt.subplot(1, 2, 2)
plt.hist(pw_durations, bins=bins_pw, color='orange', alpha=0.7)
plt.xscale('log')
plt.xlabel("Duration (s) [log scale]")
plt.ylabel("Frequency")
plt.title("Particle-Wall Collision Durations")

plt.tight_layout()
plt.show()