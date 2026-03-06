import os

import numpy as np
import pickle

class RockyDeck:
    def __init__(self, contacts_filename, particles_filename):
        self.contacts_filename = contacts_filename
        self.particles_filename = particles_filename
        
        self.n_timesteps = 0
        self.n_particles = 0

        self.p_ids = None
        self.p_coords = None
        self.p_m = None
        self.p_r = None
        self.p_i = None
        self.p_v = None
        self.p_av = None

        self.pp_c_ids = None
        self.pp_c_coords = None
        self.pp_c_fn = None
        self.pp_c_ft = None

        self.pw_c_ids = None
        self.pw_c_fn = None
        self.pw_c_ft = None

        self.p_net_forces = None
        self.p_net_torques = None

        self.pp_contact_vectors = None
        self.pp_relative_dist = None
        self.pp_dist_vector = None

        self.pp_n_loss = None
        self.pp_t_loss = None

        self.pw_n_loss = None
        self.pw_t_loss = None

        self.load_particles()
        self.load_contacts()
        self.calculate_particle_forces_torques()
        self.calculate_contact_vectors()
        self.calculate_energy_dissipation()

    def load_particles(self):
        particles_loaded = np.load(self.particles_filename, allow_pickle=True)

        self.n_timesteps = particles_loaded['Particle ID'].shape[0]
        self.n_particles = particles_loaded['Particle ID'].shape[1]
        
        self.p_ids = particles_loaded['Particle ID']
        self.p_m = particles_loaded['Particle Mass']
        self.p_r = particles_loaded['Particle Equivalent Diameter'] / 2.0
        self.p_i = (2/5) * self.p_m * (self.p_r ** 2)

        self.p_coords = np.stack((particles_loaded['Coordinate : X'], particles_loaded['Coordinate : Y'], particles_loaded['Coordinate : Z']), axis=-1)
        self.p_v = np.stack((particles_loaded['Velocity : Translational : X'], particles_loaded['Velocity : Translational : Y'], particles_loaded['Velocity : Translational : Z']), axis=-1)
        self.p_av = np.stack((particles_loaded['Velocity : Rotational : X'], particles_loaded['Velocity : Rotational : Y'], particles_loaded['Velocity : Rotational : Z']), axis=-1)

        pass

    def load_contacts(self):
        contacts_loaded = np.load(self.contacts_filename, allow_pickle=True)

        # contacts_from and contacts_to are lists of np arrays, one array per timestep
        # each array shape: (n_contacts_at_timestep,)
        # create list of shape (n_timesteps, n_contacts_at_timestep, 2) for ids

        # --- Raw contact data ---
        c_from, c_to = contacts_loaded['Particle ID From'], contacts_loaded['Triangle or Particle ID To']
        c_x, c_y, c_z = contacts_loaded['Contact : Coordinate : X'], contacts_loaded['Contact : Coordinate : Y'], contacts_loaded['Contact : Coordinate : Z']
        c_fn_x, c_fn_y, c_fn_z = contacts_loaded['Force : Normal : X'], contacts_loaded['Force : Normal : Y'], contacts_loaded['Force : Normal : Z']
        c_ft_x, c_ft_y, c_ft_z = contacts_loaded['Force : Tangential : X'], contacts_loaded['Force : Tangential : Y'], contacts_loaded['Force : Tangential : Z']

        # --- Format contact data ---
        c_ids = [ np.stack((c_from[t], c_to[t]), axis=-1) for t in range(self.n_timesteps) ]
        c_coords = [ np.stack((c_x[t], c_y[t], c_z[t]), axis=-1) for t in range(self.n_timesteps) ]
        c_fn = [ np.stack((c_fn_x[t], c_fn_y[t], c_fn_z[t]), axis=-1) for t in range(self.n_timesteps) ]
        c_ft = [ np.stack((c_ft_x[t], c_ft_y[t], c_ft_z[t]), axis=-1) for t in range(self.n_timesteps) ]
        
        # --- Split contacts into particle-particle and particle-wall ---

        # At index 0 leave empty, as there are no contacts at timestep 0, split between pp and pw contacts
        self.pp_c_ids = [np.empty((0,2), dtype=int)]
        self.pp_c_coords = [np.empty((0,3), dtype=float)]
        self.pp_c_fn = [np.empty((0,3), dtype=float)]
        self.pp_c_ft = [np.empty((0,3), dtype=float)]


        self.pw_c_ids = [np.empty((0,2), dtype=int)]
        self.pw_c_fn = [np.empty((0,3), dtype=float)]
        self.pw_c_ft = [np.empty((0,3), dtype=float)]

        # The type of contact for each pair of shape (n_timesteps, n_contacts, 1) (0 = particle-particle, 1 = particle-wall)
        contact_types = contacts_loaded['Contact Type']
        for t in range(1, self.n_timesteps):
            pp_mask = (contact_types[t] == 0)
            pw_mask = (contact_types[t] == 1)

            self.pp_c_ids.append( c_ids[t][pp_mask] )
            self.pp_c_coords.append( c_coords[t][pp_mask] )
            self.pp_c_fn.append( c_fn[t][pp_mask] )
            self.pp_c_ft.append( c_ft[t][pp_mask] )

            self.pw_c_ids.append( c_ids[t][pw_mask] )
            self.pw_c_fn.append( c_fn[t][pw_mask] )
            self.pw_c_ft.append( c_ft[t][pw_mask] )

        pass

    def calculate_particle_forces_torques(self):
        accelerations = np.zeros_like(self.p_v)
        rotational_accelerations = np.zeros_like(self.p_av)

        # Calculate accelerations
        accelerations[1:] = np.diff(self.p_v, axis=0)
        rotational_accelerations[1:] = np.diff(self.p_av, axis=0)

        # Calculate net force vectors using F = m*a
        net_forces = np.zeros_like(self.p_v)
        for t in range(self.n_timesteps):
            for p in range(self.n_particles):
                net_forces[t, p, :] = self.p_m[t, p] * accelerations[t, p, :]

        # Calculate net torque vectors using τ = I*α, I = 2/5*m*r^2
        net_torques = np.zeros_like(self.p_av)
        for t in range(self.n_timesteps):
            for p in range(self.n_particles):
                net_torques[t, p, :] = self.p_i[t, p] * rotational_accelerations[t, p, :]

        self.p_net_forces = net_forces
        self.p_net_torques = net_torques

        pass

    def calculate_contact_vectors(self):
        
        # calculate contactVector1 = contactPoint - particlePosition for each particle-particle contact
        contact_vectors = []
        for t in range(self.n_timesteps):
            contact_vectors_t = []
            for c in range(self.pp_c_ids[t].shape[0]):
                particle_id = self.pp_c_ids[t][c][0]
                particle_position = self.p_coords[t][particle_id]
                contact_point = self.pp_c_coords[t][c]
                contact_vector = contact_point - particle_position
                contact_vectors_t.append(contact_vector)
            contact_vectors.append( np.array(contact_vectors_t) )

        
        self.pp_contact_vectors = contact_vectors
        self.pp_relative_dist = [ np.linalg.norm(self.pp_contact_vectors[t], axis=1) if self.pp_contact_vectors[t].shape[0] > 0 else np.array([]) for t in range(self.n_timesteps) ]
        self.pp_dist_vector = [ self.pp_contact_vectors[t] * 2.0 for t in range(0, self.n_timesteps) ]

        pass

    def calculate_energy_dissipation(self):
        particles_loaded = np.load(self.particles_filename, allow_pickle=True)

        # Arrays of size (n_timesteps) with total energy dissipation at each timestep
        pp_dissipation = particles_loaded['Energy : Dissipation | My Particle X My Particle']
        pw_dissipation = particles_loaded['Energy : Dissipation | My Particle X DyAoR_Cylinder']

        # pp normal and tangential loss are half of total pp dissipation each
        self.pp_n_loss = pp_dissipation / 2.0
        self.pp_t_loss = pp_dissipation / 2.0

        self.pw_n_loss = pw_dissipation / 2.0
        self.pw_t_loss = pw_dissipation / 2.0

        pass

# Create the deck and save it as a pickle file if it doesn't exist yet
if os.path.exists("./data-drum/1-staging/rocky_deck.pkl"):
    print("Rocky deck already exists, skipping creation.")
else:
    print("Creating rocky deck from raw data...")
    deck = RockyDeck("./data-drum/0-raw/contacts.npz", "./data-drum/0-raw/particles.npz")

    # save the deck as pkl file
    with open("./data-drum/1-staging/rocky_deck.pkl", "wb") as f:
        pickle.dump(deck, f)