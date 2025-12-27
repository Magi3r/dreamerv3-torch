import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier

class EpisodicMemory(Dataset):
    """
    Episodic Memory object for trajectory management.

    general idea: store overlapping trajectories inside same object.
                  keep dictionary of (z,h,a): (trajectoryobject, nr, uncertainty)
                    nr stores the number within trajectory object
                ...
    """
    def __init__(self, trajectory_length: int, uncertainty_threshold: float, z_shape, action_shape, k_nn: int = 5):
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

        self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryMemory, idx, uncertainty)

        self.current_trajectory: TrajectoryObject | None = None

        self.prev_state = None

    def __len__(self):
        return len(self.trajectories)

    def __create_traj(self, key: tuple, uncertainty: float):
        """ Create new trajectory 
            Create an empty trajectory, that is accessible by a key 
        """
        # key: (h_t, z_t, a_t)
        # uncertainty: float
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length) # z_shape, action_shape
        self.current_trajectory = trajectory

        # self.trajectories[key] = (trajectory, trajectory.last_idx(), uncertainty)
        self.trajectories[key] = (trajectory, trajectory.new_traj(), uncertainty)
        
    def __fill_traj(self, value):
        """ Adds a new transition (z, a) to the current trajectory. If the trajectory becomes full, it clears current_trajectory."""
        # value: (z_{t'}, a_{t'})
        assert(self.current_trajectory is not None)
        if self.current_trajectory.add(value) == 0:
            self.current_trajectory = None

    def remove_traj(self, key: tuple):
        """ Remove trajectory by its key """
        assert(key in self.trajectories)
        traj_obj, idx, _ = self.trajectories[key]
        traj_obj.del_traj(idx)
        del self.trajectories[key]

    def step(self, state: dict, action, uncertainty: float, done:bool=False):
        """ Step through the memory with new transition 
        called each step :)

        Manages the episodic memory (build it with incoming values[state, action, uncertainty]).
        needs to store previous state, as it will be the key if current state is uncertain.
        -----

        Start a new trajectory if uncertainty exceeds threshold.
        Fill the current trajectory with previous (state, action)
        Ends the trajectory and clears bookkeeping if done=True.
        """
        #initial step, just store. Even if uncertain dont have a key for it
        if self.prev_state == None:
            self.prev_state = state
            return
        
        # add new trajecory
        if uncertainty > self.uncertainty_threshold:
            assert(self.prev_state is not None)
            # value = (z, h)          # (z_t, h_t)
            key = (self.prev_state["deter"], self.prev_state["stoch"], action)    # (z_t, h_t, a_t) 
            
            if self.current_trajectory is not None:
                self.__fill_traj((self.prev_state["stoch"], action))
            self.__create_traj(key, uncertainty)
        # just fill trajectory space
        elif self.current_trajectory is not None:
            assert(self.prev_state is not None)
            
            self.__fill_traj((self.prev_state["stoch"], action))
        # no space: no trajectory
        else:
            self.current_trajectory = None

        # manage final (done) step:
        # clear state and current trajectory as next step will be totally independend from this
        if done:
            if self.current_trajectory is not None:
                self.__fill_traj((state["stoch"], None))
            self.current_trajectory = None
            self.prev_state = None
        else:
            self.prev_state = state #.copy()?

    def flatten_key(self, key):
        """Convert (z, h, a) tensors into one numpy vector."""
        z, h, a = key  # each is a torch tensor
        z = z.flatten().cpu().numpy()
        h = h.flatten().cpu().numpy()
        a = a.flatten().cpu().numpy()
        return np.concatenate([z, h, a], axis=0)
    
    def __str__(self):
        return f"EM| Num trajectories: {len(self.trajectories)}\
            | Trajectory length: {self.trajectory_length}\
                | Uncertainty thr.: {self.uncertainty_threshold}\
                    \n          | Current trajectory: {self.current_trajectory}"


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

    # def add(self, key: tuple, value: tuple, uncertainty: float):
    #     """ Add new trajectory """
    #     # key: (h_t, z_t, a_t)
    #     # value: (z_{t'}, a_{t'})
    #     # uncertainty: float
    #     trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length) # z_shape, action_shape

    #     trajectory.add(value)

    def kNN(self, key: tuple, k: int = 1):
        """Return the k-nearest neighbors among stored trajectory keys."""
        
        def flatten_key(k):
            z, h, a = k  # each is a torch tensor
            z = z.detach().cpu().numpy().flatten()
            h = h.detach().cpu().numpy().flatten()
            a = a.detach().cpu().numpy().flatten()

            res = np.concatenate([z, h, a])
            return res
        
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        key_array = list(self.trajectories.keys())
        assert(len(key_array) > 0), "No trajectories stored in EpisodicMemory."
        search_space = np.stack([flatten_key(key) for key in key_array])  # shape: (N, D)

        x = flatten_key(key).reshape(1, -1)
        knn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"  # or cosine, mahalanobis, etc.
        ).fit(search_space)

        distances, indices = knn.kneighbors(x) # [None]
        
        # Return the actual trajectory objects
        keys = list(self.trajectories.keys())
        neighbors = [self.trajectories[keys[i]] for i in indices[0]]

        return neighbors
        

    # def kNN(self, key, k:int=1):
    #     # key: (h_t, z_t, a_t)
    #     Warning("For now ignore z_t.")

    #     # (h_t, z_t, a_t) -> (h_t, z_t)
    #     # entries = np.array([[k[0], k[1]] for k in self.trajectories.keys()])


    #     # return k nearest neighbors based on some distance metric
    #     raise NotImplementedError("kNN method not implemented yet.")
    
class TrajectoryObject:
    def __init__(self, trajectory_length: int):
        self.trajectory_length: int = trajectory_length
        self.free_space: int = trajectory_length

        self.num_trajectories : int = 0
        self.trajectory_memory: dict = {}
        self.current_trajectory_id: int = 0

    def new_traj(self):
        nr_idx = self.num_trajectories

        self.trajectory_memory[nr_idx] = np.empty(self.trajectory_length, dtype=object)
        self.free_space = self.trajectory_length

        self.current_trajectory_id = nr_idx
        self.num_trajectories += 1
        return nr_idx

    def del_traj(self, traj_nr):
        # The fist version should be minimally faster, but we leave things up to the garbage colelctor, which might not be such a good idea.
        
        self.trajectory_memory[traj_nr] = None
        # del(self.trajectory_memory[traj_nr])

    def add(self, value: tuple) -> int:
        self.trajectory_memory[self.current_trajectory_id][-self.free_space] = value
        self.free_space -= 1

        return self.free_space
    

    def memory(self):
        return self.trajectory_memory[self.current_trajectory_id]
        """
        memory = self.trajectory_memory[0]
        for i in range(1,self.num_trajectories):
            memory = np.concatenate(memory, self.trajectory_memory[i])
        return memory
        """


    def __str__(self):
        return f"TrajectoryObj| Free space: {self.free_space}| Trajectory length: {self.trajectory_length}"
