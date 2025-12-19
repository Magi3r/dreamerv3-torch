import os
import sys

# PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/my_project
# sys.path.append(str(PROJECT_ROOT))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from episodic_memory import EpisodicMemory as EM
import numpy as np
import torch

def test_episodic_memory_basic():
    print("Running EpisodicMemory basic test...")

    # ---- setup ----
    traj_len = 3
    uncertainty_threshold = 0.5

    mem = EM(
        trajectory_length=traj_len,
        uncertainty_threshold=uncertainty_threshold,
        z_shape=(2,),
        action_shape=(1,),
    )

    # Dummy states & actions
    def make_state(v):
        return {
            "deter": torch.tensor([v, v + 1.0]),
            "stoch": torch.tensor([v + 2.0, v + 3.0]),
        }

    action0 = torch.tensor([0.0])
    action1 = torch.tensor([1.0])

    # ---- step 1: no trajectory yet ----
    mem.step(make_state(0.0), action0, uncertainty=0.1)
    assert mem.current_trajectory is None
    assert len(mem.trajectories) == 0

    # ---- step 2: high uncertainty → start trajectory ----
    mem.step(make_state(1.0), action0, uncertainty=0.9)
    print(("Mem: ", mem.__str__()))
    print(("trajectories mem: ", mem.trajectories))
    assert mem.current_trajectory is not None
    assert len(mem.trajectories) == 1

    traj_obj = mem.current_trajectory
    assert traj_obj.free_space == traj_len

    # ---- step 3: fill trajectory ----
    mem.step(make_state(2.0), action1, uncertainty=0.1)
    assert traj_obj.free_space == traj_len - 1

    mem.step(make_state(3.0), action1, uncertainty=0.1)
    assert traj_obj.free_space == traj_len - 2

    # ---- step 4: end trajectory ----
    print(("trajectories mem free_space: ", list(mem.trajectories.values())[0][0].free_space))
    mem.step(make_state(4.0), action1, uncertainty=0.1, done=True)
    print(("Mem: ", mem.__str__()))
    print(("trajectories mem: ", mem.trajectories))
    print(("trajectories mem memory: ", list(mem.trajectories.values())[0][0].memory))
    print(("trajectories mem free_space: ", list(mem.trajectories.values())[0][0].free_space))
    assert mem.current_trajectory is None
    assert mem.prev_state is None

    # ---- checks ----
    trajs = list(mem.trajectories.values())
    stored_traj, offset, unc = trajs[0]

    print("Stored trajectory memory:", stored_traj.memory)
    print("Free space:", stored_traj.free_space)
    print("Offset:", offset)
    print("Uncertainty:", unc)

    assert stored_traj.free_space == traj_len - 3 or stored_traj.free_space == 0

    print("✅ EpisodicMemory basic test passed.")

def test_multiple_trajectories_two_object():
    print("\nRunning multiple-trajectory test...")

    traj_len = 4
    uncertainty_threshold = 0.5

    mem = EM(
        trajectory_length=traj_len,
        uncertainty_threshold=uncertainty_threshold,
        z_shape=(2,),
        action_shape=(1,),
    )

    def make_state(v):
        return {
            "deter": torch.tensor([42, v]),
            "stoch": torch.tensor([v, v]),
        }

    action = torch.tensor([1.0])

    # ---- trajectory 1 (object 1) ----
    mem.step(make_state(0.0), action, uncertainty=0.1)  # start
    print("LEN ALL TRAJdfgd gf :", len(mem.trajectories))
    assert len(mem.trajectories) == 0
    mem.step(make_state(1.0), action, uncertainty=0.9)
    mem.step(make_state(2.0), action, uncertainty=0.1, done=True)

    trajs = list(mem.trajectories.values())
    stored_traj, offset, unc = trajs[0]
    print("Stored trajectory memory:", stored_traj.memory)
    print("Free space:", stored_traj.free_space)
    print("Offset:", offset)
    print("Uncertainty:", unc)
    print("----\n----")

    assert len(mem.trajectories) == 1

    traj_obj_1, idx_1, _ = list(mem.trajectories.values())[0]
    assert idx_1 == 0
    assert traj_obj_1.num_trajectories == 1

    # ---- trajectory 1 (object 2) ----
    mem.step(make_state(3.0), action, uncertainty=0.1)  # start new

    mem.step(make_state(4.0), action, uncertainty=0.9)  
    trajs = list(mem.trajectories.values())[1][0]

    mem.step(make_state(5.0), action, uncertainty=0.1)
    trajs = list(mem.trajectories.values())[1][0]

    # trajs = list(mem.trajectories.values())[0][0]
    # print("Free space original traj:", trajs.free_space)

    mem.step(make_state(6.0), action, uncertainty=0.9)  
    trajs = list(mem.trajectories.values())[1][0]

    mem.step(make_state(7.0), action, uncertainty=0.1)
    trajs = list(mem.trajectories.values())[1][0]

    mem.step(make_state(8.0), action, uncertainty=0.1, done=True)
    trajs = list(mem.trajectories.values())[1][0]

    trajs = list(mem.trajectories.values())
    trajs_obs = list(mem.trajectories.keys())
    i=0
    for stored_traj, idx, unc in trajs:
        print(f"Entry:{i} Key:", trajs_obs[i])
        print("Stored trajectory memory:", stored_traj.memory)
        print("Free space:", stored_traj.free_space)
        print(f"Index: {idx} -> offset: {stored_traj.traj_num_to_offset[idx]}")
        print("Uncertainty:", unc)
        i += 1

    assert len(mem.trajectories) == 3

    traj_obj_2, idx_2, _ = list(mem.trajectories.values())[2]
    assert traj_obj_2 is not traj_obj_1          # SAME object
    assert traj_obj_2.traj_num_to_offset[1] == 2
    assert traj_obj_2.num_trajectories == 2

    # ---- inspect memory layout ----
    mem_array = traj_obj_2.memory

    print("Full memory:", mem_array)
    print("Trajectory start indices:", traj_obj_2.traj_num_to_offset[:2])

    # Trajectory 1 occupies indices [0, 1, 2]
    t1_start = traj_obj_2.traj_num_to_offset[0]
    t2_start = traj_obj_2.traj_num_to_offset[1]

    assert t1_start == 0
    assert t2_start == 2

    # Values should be tuples
    assert mem_array[0] is not None
    assert isinstance(mem_array[0], tuple)

    print("✅ Multiple trajectories stored in one object correctly.")

def test_multiple_trajectories_single_object_deletion():
    print("\nRunning multiple-trajectory test + deletion...")

    traj_len = 4
    uncertainty_threshold = 0.5

    mem = EM(
        trajectory_length=traj_len,
        uncertainty_threshold=uncertainty_threshold,
        z_shape=(2,),
        action_shape=(1,),
    )

    def make_state(v):
        return {
            "deter": torch.tensor([42, v]),
            "stoch": torch.tensor([v, v]),
        }

    action = torch.tensor([1.0])

    # ---- trajectory 1 (object 1) ----
    mem.step(make_state(0.0), action, uncertainty=0.1)  # start
    mem.step(make_state(1.0), action, uncertainty=0.9)
    
    assert len(mem.trajectories) == 1

    mem.step(make_state(2.0), action, uncertainty=0.9)
    mem.step(make_state(3.0), action, uncertainty=0.1) # ! this will not be added here

    assert len(mem.trajectories) == 2

    trajs = list(mem.trajectories.values())
    stored_traj, offset, unc = trajs[0]
    print("Stored trajectory memory:", stored_traj.memory)
    print("Free space:", stored_traj.free_space)
    # print("Offset:", offset)
    # print("Uncertainty:", unc)

    traj_obj1, idx1, _ = list(mem.trajectories.values())[0]
    print(traj_obj1)

    first_key = list(mem.trajectories.keys())[0]
    mem.remove_traj(first_key)
    print("removes first traj with value: ", mem.trajectories[first_key][0])

    mem.step(make_state(4.0), action, uncertainty=0.9)

    traj_obj1, idx1, _ = list(mem.trajectories.values())[0]
    print(traj_obj1)

    print("\n\nNow insert 3 new trajs with some gabs to test deletion...\n")

    mem = EM(
        trajectory_length=traj_len,
        uncertainty_threshold=uncertainty_threshold,
        z_shape=(2,),
        action_shape=(1,),
    )
    # ---- trajectory 1 (object 1) ----
    mem.step(make_state(0.0), action, uncertainty=0.1)  # start
    mem.step(make_state(1.0), action, uncertainty=0.9)

    assert traj_obj1.free_space==0
    assert traj_obj1 is None
    assert len(mem.trajectories) == 0

    return
    traj_obj_1, idx_1, _ = list(mem.trajectories.values())[0]
    assert idx_1 == 0
    assert traj_obj_1.num_trajectories == 1

    # ---- trajectory 1 (object 2) ----
    mem.step(make_state(3.0), action, uncertainty=0.1)  # start new
    print("LEN ALL TRAJ :", len(mem.trajectories))    

    mem.step(make_state(4.0), action, uncertainty=0.9)  
    trajs = list(mem.trajectories.values())[1][0]
    print("LEN ALL TRAJ:", len(mem.trajectories))
    print("Free space 1:", trajs.free_space)

    mem.step(make_state(5.0), action, uncertainty=0.1)
    trajs = list(mem.trajectories.values())[1][0]
    print("LEN ALL TRAJ:", len(mem.trajectories))
    print("Free space 2:", trajs.free_space)

    # trajs = list(mem.trajectories.values())[0][0]
    # print("Free space original traj:", trajs.free_space)

    mem.step(make_state(6.0), action, uncertainty=0.9)  
    trajs = list(mem.trajectories.values())[1][0]
    print("LEN ALL TRAJ:", len(mem.trajectories))
    print("Free space 3:", trajs.free_space)

    mem.step(make_state(7.0), action, uncertainty=0.1)
    trajs = list(mem.trajectories.values())[1][0]
    print("LEN ALL TRAJ:", len(mem.trajectories))
    print("Free space 4:", trajs.free_space)

    mem.step(make_state(8.0), action, uncertainty=0.1, done=True)
    trajs = list(mem.trajectories.values())[1][0]
    print("LEN ALL TRAJ:", len(mem.trajectories))
    print("Free space 5:", trajs.free_space)

    trajs = list(mem.trajectories.values())
    trajs_obs = list(mem.trajectories.keys())
    i=0
    for stored_traj, idx, unc in trajs:
        print(f"Entry:{i} Key:", trajs_obs[i])
        print("Stored trajectory memory:", stored_traj.memory)
        print("Free space:", stored_traj.free_space)
        print("Index:", idx)
        print("Uncertainty:", unc)
        i += 1
        # print("\n\n")
        # print(stored_traj)
        # print("\n\n")

    assert len(mem.trajectories) == 3

    traj_obj_2, idx_2, _ = list(mem.trajectories.values())[2]
    assert traj_obj_2 is not traj_obj_1          # SAME object
    assert traj_obj_2.traj_num_to_offset[1] == 2
    assert traj_obj_2.num_trajectories == 2

    # ---- inspect memory layout ----
    mem_array = traj_obj_2.memory

    print("Full memory:", mem_array)
    print("Trajectory start indices:", traj_obj_2.traj_num_to_offset[:2])

    # Trajectory 1 occupies indices [0, 1, 2]
    t1_start = traj_obj_2.traj_num_to_offset[0]
    t2_start = traj_obj_2.traj_num_to_offset[1]

    assert t1_start == 0
    assert t2_start == 2

    # Values should be tuples
    assert mem_array[0] is not None
    assert isinstance(mem_array[0], tuple)

    print("✅ Multiple trajectories stored in one object correctly.")

if __name__ == "__main__":
    # test_episodic_memory_basic()

    # test_multiple_trajectories_two_object()

    test_multiple_trajectories_single_object_deletion()



# import numpy as np
# import torch

# # Mock state with deterministic and stochastic parts
# def make_state(deter_val, stoch_val):
#     return {"deter": torch.tensor([deter_val]), "stoch": torch.tensor([stoch_val])}

# def test_episodic_memory():
#     trajectory_length = 5
#     uncertainty_threshold = 0.5

#     em = EpisodicMemory(
#         trajectory_length=trajectory_length,
#         uncertainty_threshold=uncertainty_threshold,
#         z_shape=(1,),
#         action_shape=(1,),
#         k_nn=2
#     )

#     # Step 1: low uncertainty -> should not create trajectory
#     state1 = make_state(1, 10)
#     em.step(state1, action=0, uncertainty=0.1)
#     assert len(em.trajectories) == 0, "No trajectory should be created yet."

#     # Step 2: high uncertainty -> create trajectory from prev state
#     state2 = make_state(2, 20)
#     em.step(state2, action=1, uncertainty=0.6)
#     assert len(em.trajectories) == 1, "A trajectory should be created."
#     traj1 = list(em.trajectories.values())[0][0]
#     assert traj1.memory[0] == (state1["stoch"], 1), "Trajectory should start with previous state and action."

#     # Step 3: low uncertainty -> fill trajectory
#     state3 = make_state(3, 30)
#     em.step(state3, action=2, uncertainty=0.1)
#     assert traj1.memory[1] == (state2["stoch"], 2), "Trajectory should contain second transition."

#     # Step 4: another high uncertainty nearby -> should continue the same trajectory
#     state4 = make_state(4, 40)
#     em.step(state4, action=3, uncertainty=0.7)
#     # Should still have 1 trajectory object
#     assert len(em.trajectories) == 1, "Overlapping uncertain transitions should stay in same trajectory."
#     assert traj1.memory[2] == (state3["stoch"], 3), "Trajectory continues with previous state and action."

#     # Step 5: high uncertainty far apart -> new trajectory
#     # Reset memory to simulate distant state
#     em.prev_state = make_state(100, 1000)
#     em.current_trajectory = None
#     state5 = make_state(101, 1010)
#     em.step(state5, action=5, uncertainty=0.8)
#     assert len(em.trajectories) == 2, "Non-overlapping uncertain transition should create new trajectory."

#     print("All tests passed!")

# test_episodic_memory()