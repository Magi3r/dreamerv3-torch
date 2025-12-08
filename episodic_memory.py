import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier

class EpisodicMemory(Dataset):
    """
    Episodic Memory object for trajectory management.
    """
    def __init__(self, trajectory_length: int, uncertainty_threshold: float = 0.0):
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

    def add(self, key, value, uncertainty):
        # key: (h_t, z_t, a_t)
        # value: (z_{t'}, a_{t'})
        # uncertainty: float
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length)
        
        trajectory.add(value, is_new_trajectory=True)
    
    def fill_traj(self, value):
        assert(self.current_trajectory is not None)
        self.current_trajectory.add(value, is_new_trajectory=False)

    def step(self, state, action, uncertainty, done:bool=False):
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
        self.memory: np.array = np.zeros((trajectory_length,))
        """"The actual trajectories."""

    def add(self, value, is_new_trajectory: bool):
        """
        Add a value into the trajectory. Aut
                
        :param value: The value to add.
        :param is_new_trajectory: Whether a new trajectory starts here.
        :type is_new_trajectory: bool
        """
        if is_new_trajectory:
            self.memory = np.concat(self.memory, np.zeros(self.trajectory_length-self.free_space,))
            self.free_space = self.trajectory_length
        self.memory[-self.free_space]=value
        self.free_space -= 1

    def last_idx(self):
        return self.memory.shape[0] - self.free_space








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