import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier

class EpisodicMemory(Dataset):
    """
    Episodic Memory object for trajectory management.
    """
    def __init__(self, trajectory_length: int, uncertainty_threshold: float, z_shape, h_shape, action_shape, k_nn: int = 5):
        """
        Docstring for __init__
        
        :param self: Description
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        :param uncertainty_threshold: Threshold for uncertainty to start a new trajectory
        :type uncertainty_threshold: float
        """
        self.trajectory_length: int = trajectory_length
        self.uncertainty_threshold: float = uncertainty_threshold
        """Threshold for uncertainty to start a new trajectory"""
        self.key_size = z_shape 

        self.k_nn: int = k_nn
        """Number of nearest neighbors to retrieve."""

        self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryMemory, offset, uncertainty)

        self.current_trajectory: TrajectoryObject | None = None
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        Exception("Not implemented yet.")
        # mem, offset, _ = self.trajectories[key]
        # sample = mem.get_trajectory(offset)
        # return sample
    
    def get_data_loader(self) -> DataLoader: # does this make sense? ~ Jannek: ¯\_(ツ)_/¯
        return DataLoader(self, batch_size=1, shuffle=False)
    
    # this would ne similar to their train and eval approach for Dreamer agent
    def get_data_generator(self, batch_length: int, seed: int = 42):
        # np_random = np.random.RandomState(seed)

        # # !! this code still does random sampling, not simple iteration over items !!
        # while True:
        #     # If there are no stored trajectories we cannot sample anything.
        #     # Yield an empty dict and continue; the caller can decide how to handle  this situation (e.g. skip a training step).
        #     if len(self.trajectories) == 0:
        #         yield {}
        #         continue

        #     size = 0                # how many timesteps we have collected so far
        #     batch = None            # the dict that will hold the concatenated data

        #     # Uniform sampling probabilities – replace with a smarter scheme later.
        #     traj_keys = list(self.trajectories.keys())
        #     p = np.ones(len(traj_keys), dtype=np.float32)
        #     p /= p.sum()

        #     # 4️⃣  Keep pulling trajectories until we have at least `batch_length`
        #     while size < batch_length:
        #         # 4.1️⃣  Choose a trajectory (with replacement)
        #         chosen_key = np_random.choice(traj_keys, p=p)
        #         traj_obj, offset, _ = self.trajectories[chosen_key]

        #         # 4.2️⃣  Extract the raw memory stored in the trajectory.
        #         #        We expect `traj_obj.memory` to be a dict of NumPy arrays
        #         #        (e.g. {"z": ..., "h": ..., "action": ..., "reward": ...}).
        #         assert(traj_obj is not None)
        #         assert(isinstance(traj_obj.memory, np.array))
        #         mem: np.array = traj_obj.memory

        #         # 4.3️⃣  Determine the length of the trajectory.
        #         total_len = len(next(iter(mem)))

        #         # Skip trajectories that are too short to provide at least one
        #         # transition (the original code requires `total >= 2`).
        #         if total_len < 2:
        #             continue

        #         # 4.4️⃣  First slice of the batch (or initialise it)
        #         if batch is None:
        #             # Random start index inside the chosen trajectory.
        #             start_idx = int(np_random.randint(0, total_len - 1))
        #             end_idx = min(start_idx + batch_length, total_len)

        #             batch = {
        #                 mem[start_idx:end_idx].copy()
        #             }

        #             # Mark the first transition of the newly‑created batch.
        #             if "is_first" in batch:
        #                 batch["is_first"][0] = True

        #         # 4.5️⃣  Subsequent slices – we always continue from the *beginning*
        #         #       of a trajectory (index 0) because the original Dreamer code
        #         #       does exactly that when it needs to fill the remaining space.
        #         else:
        #             # How many more timesteps do we still need?
        #             needed = batch_length - size
        #             # We always start from the beginning of the trajectory.
        #             start_idx = 0
        #             end_idx = min(start_idx + needed, total_len)

        #             # Append the new slice to the existing batch.
        #             batch = {
        #                 k: np.append(batch[k],
        #                              v[start_idx:end_idx].copy(),
        #                              axis=0)
        #                 for k, v in mem_dict.items()
        #                 if "log_" not in k
        #             }

        #             # Fix the ``is_first`` flag for the newly‑added segment.
        #             if "is_first" in batch:
        #                 batch["is_first"][size] = True

        #         # 4.6️⃣  Update bookkeeping
        #         size = len(next(iter(batch.values())))

        #     yield batch
        pass
        #### How they have dont it: ####
        # episodes -> <class 'collections.OrderedDict'>
        #   keys: file_names
        #   values: <class 'dict'> | len = 7
        # ... see tools.py

    def add(self, key: tuple, value: tuple, uncertainty: float):
        """ Add new trajectory """
        # key: (h_t, z_t, a_t)
        # value: (z_{t'}, a_{t'})
        # uncertainty: float
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length)
        
        trajectory.add(value, is_new_trajectory=True)
    
    def fill_traj(self, value):
        """ Add entry to existing trajectory """
        assert(self.current_trajectory is not None)
        self.current_trajectory.add(value, is_new_trajectory=False)

    def step(self, state: dict, action, uncertainty: float, done:bool=False):
        if uncertainty > self.uncertainty_threshold:
            # add new trajecory
            z = state["stoch"]
            h = state["deter"]
            value = (z, h) # (z_t, h_t)
            key = (z, h, action) # (z_t, h_t, a_t) 
            self.add(key, value, uncertainty)
        elif self.current_trajectory is not None and self.current_trajectory.free_space > 0:
            # just fill trajectory space
            value = (state["stoch"], state["deter"])# (z_t, h_t)
            self.fill_traj(value)
        else:
            self.current_trajectory = None

    def flatten_key(self, key):
        """Convert (z, h, a) tensors into one numpy vector."""
        z, h, a = key  # each is a torch tensor
        z = z.flatten().cpu().numpy()
        h = h.flatten().cpu().numpy()
        a = a.flatten().cpu().numpy()
        return np.concatenate([z, h, a], axis=0)
    
    def __str__(self):
        return f"EM| Num trajectories: {len(self.trajectories)}| Trajectory length: {self.trajectory_length}| Uncertainty thr.: {self.uncertainty_threshold}\n          | Current trajectory: {self.current_trajectory}"

    def kNN(self, key, k: int = 1):
        """Return the k-nearest neighbors among stored trajectory keys."""
        if len(self.trajectories) == 0:
            return []

        # Convert all existing keys → flattened vectors
        X = np.array([self.flatten_key(k_) for k_ in self.trajectories.keys()])

        # Convert query key → flattened vector
        query = self.flatten_key(key).reshape(1, -1)

        # Dummy labels (we only care about distances)
        y = np.arange(len(X))

        # Build kNN model
        knn = KNeighborsClassifier(n_neighbors=min(k, len(X)), metric='euclidean')
        knn.fit(X, y)

        # Get k nearest neighbors
        neighbor_idxs = knn.kneighbors(query, return_distance=False)[0]

        # Return the actual trajectory objects
        keys = list(self.trajectories.keys())
        neighbors = [self.trajectories[keys[i]] for i in neighbor_idxs]

        return neighbors

    # def kNN(self, key, k:int=1):
    #     # key: (h_t, z_t, a_t)
    #     Warning("For now ignore z_t.")

    #     # (h_t, z_t, a_t) -> (h_t, z_t)
    #     # entries = np.array([[k[0], k[1]] for k in self.trajectories.keys()])


    #     # return k nearest neighbors based on some distance metric
    #     raise NotImplementedError("kNN method not implemented yet.")
    
class TrajectoryObject:
    """
    Memory object for trajectories
    """
    def __init__(self, trajectory_length: int):
        """
        Docstring for __init__
        
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        """
        self.trajectory_length: int = trajectory_length
        """The maximum length each trajectory will have."""
        self.free_space: int = trajectory_length
        """How much free space this object still has. Always resets if new trajectory starts."""
        self.memory: np.array = np.zeros((trajectory_length,))  # TODO: add size of tuple (z_t', a_t')
        """"The actual trajectories."""

    def add(self, value: tuple, is_new_trajectory: bool):
        """
        Add a value into the trajectory. Aut
                
        :param value: The value to add.
        :param is_new_trajectory: Whether a new trajectory starts here.
        :type is_new_trajectory: bool
        """
        if is_new_trajectory:
            self.memory = np.concat(self.memory, np.zeros(self.trajectory_length-self.free_space,))
            self.free_space = self.trajectory_length
        self.memory[-self.free_space]=value # TODO: add tuple (z_t', a_t')
        self.free_space -= 1

    def last_idx(self):
        return self.memory.shape[0] - self.free_space

    def __str__(self):
        return f"TrajectoryObj| Free space: {len(self.free_space)}| Trajectory length: {self.trajectory_length}"






import numpy as np

class HybridKNN:
    def __init__(self, latent_tuples, w_discrete=1.0, w_cont=1.0, include_a=True):
        """
        latent_tuples: list of tuples (z, h, a)
            z: discrete latent [num_latent_dims, num_categories] (one-hot)
            h: continuous hidden state vector
            a: discrete action one-hot vector
        w_discrete: weight for discrete Hamming distance
        w_cont: weight for continuous Euclidean distance
        include_a: whether to include action in distance
        """
        self.w_discrete = w_discrete
        self.w_cont = w_cont
        self.include_a = include_a
        
        # Stack discrete latents and actions
        self.Z = np.array([z.ravel() for z, _, a in latent_tuples])
        if include_a:
            self.A = np.array([a.ravel() for _, _, a in latent_tuples])
        else:
            self.A = None
        
        # Stack continuous hidden states
        self.H = np.array([h for _, h, _ in latent_tuples])

    def query(self, query_tuple, k=5):
        zq, hq, aq = query_tuple
        zq_flat = zq.ravel()
        hq_flat = hq.ravel()
        if self.include_a and aq is not None:
            aq_flat = aq.ravel()
        else:
            aq_flat = None
        
        # --- Hamming distance for discrete latents ---
        hz = np.mean(self.Z != zq_flat, axis=1)  # [num_samples]
        
        if self.include_a and aq_flat is not None:
            ha = np.mean(self.A != aq_flat, axis=1)
            hamming_dist = hz + ha
        else:
            hamming_dist = hz
        
        # --- Euclidean distance for continuous h ---
        cont_dist = np.linalg.norm(self.H - hq_flat, axis=1)
        
        # --- Hybrid distance ---
        dist = self.w_discrete * hamming_dist + self.w_cont * cont_dist
        
        # --- kNN ---
        idxs = np.argsort(dist)[:k]
        return idxs, dist[idxs]





import torch

class HybridKNNTorch:
    def __init__(self, latent_tuples, device='cuda', w_discrete=1.0, w_cont=1.0, include_a=True):
        """
        latent_tuples: list of tuples (z, h, a)
            z: [num_latent_dims, num_categories] one-hot
            h: continuous hidden state vector
            a: one-hot action
        device: 'cuda' or 'cpu'
        w_discrete: weight for discrete Hamming distance
        w_cont: weight for continuous Euclidean distance
        include_a: whether to include actions in distance
        """
        self.device = device
        self.w_discrete = w_discrete
        self.w_cont = w_cont
        self.include_a = include_a

        # Stack and move to device
        self.Z = torch.stack([torch.tensor(z, dtype=torch.float32) for z, _, _ in latent_tuples]).to(device)  # [N, z_dim, num_classes]
        if include_a:
            self.A = torch.stack([torch.tensor(a, dtype=torch.float32) for _, _, a in latent_tuples]).to(device)
        else:
            self.A = None
        self.H = torch.stack([torch.tensor(h, dtype=torch.float32) for _, h, _ in latent_tuples]).to(device)

        # Flatten discrete tensors for distance computation
        self.Z_flat = self.Z.flatten(start_dim=1)  # [N, z_dim*num_classes]
        if self.include_a and self.A is not None:
            self.A_flat = self.A.flatten(start_dim=1)  # [N, a_dim]

    def query(self, query_tuple, k=5):
        zq, hq, aq = query_tuple
        zq = torch.tensor(zq, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)  # [1, z_dim*num_classes]
        hq = torch.tensor(hq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, h_dim]
        if self.include_a and aq is not None:
            aq = torch.tensor(aq, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)  # [1, a_dim]

        # --- Hamming distance for discrete parts ---
        hz = (self.Z_flat != zq).float().mean(dim=1)  # [N]
        if self.include_a and aq is not None:
            ha = (self.A_flat != aq).float().mean(dim=1)
            hamming_dist = hz + ha
        else:
            hamming_dist = hz

        # --- Euclidean distance for continuous h ---
        cont_dist = torch.norm(self.H - hq, dim=1)  # [N]

        # --- Hybrid distance ---
        dist = self.w_discrete * hamming_dist + self.w_cont * cont_dist

        # --- kNN ---
        distances, indices = torch.topk(dist, k=k, largest=False)
        return indices.cpu().numpy(), distances.cpu().numpy()