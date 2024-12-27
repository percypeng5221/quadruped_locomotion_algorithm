from copy import deepcopy
import numpy as np
from utils import read_json_file

class AStarObjectMover:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state  # {object_id: position}
        self.goal_state = goal_state        # {object_id: position}

    def heuristic(self, state):
        """
        Heuristic: Sum of Euclidean distances between current and goal positions of all objects.
        """
        total_distance = 0
        for obj_id in state:
            current_pos = state[obj_id]
            goal_pos = self.goal_state[obj_id]
            distance = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
            total_distance += distance
        return total_distance

    def get_valid_actions(self, state):
        """
        Generate valid move actions based on the current state.
        For simplicity, we assume the robot can move any object directly to its goal position.
        """
        actions = []
        for obj_id in state:
            if state[obj_id] != self.goal_state[obj_id]:
                actions.append(('Move', obj_id, self.goal_state[obj_id]))
        return actions

    def apply_action(self, state, action):
        """
        Apply an action to create a new state.
        """
        new_state = deepcopy(state)
        action_type, obj_id, target_pos = action

        if action_type == 'Move':
            new_state[obj_id] = target_pos  # Move the object to the target position

        return new_state

    def a_star_search(self):
        """
        A* search to find the optimal sequence of actions.
        """
        open_set = [(self.initial_state, [], 0)]  # (state, path, g_score)
        g_score = {self.state_to_tuple(self.initial_state): 0}
        visited = set()

        while open_set:
            # Select the node with the lowest f_score = g_score + heuristic
            open_set.sort(key=lambda x: x[2] + self.heuristic(x[0]))
            current_state, path, current_g_score = open_set.pop(0)
            state_key = self.state_to_tuple(current_state)

            if state_key in visited:
                continue
            visited.add(state_key)

            if self.is_goal(current_state):
                return path  # Return the sequence of actions

            for action in self.get_valid_actions(current_state):
                new_state = self.apply_action(current_state, action)
                new_state_key = self.state_to_tuple(new_state)
                # The cost to move an object is the distance moved
                obj_id = action[1]
                current_pos = current_state[obj_id]
                target_pos = new_state[obj_id]
                movement_cost = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                tentative_g_score = current_g_score + movement_cost

                if tentative_g_score < g_score.get(new_state_key, float('inf')):
                    g_score[new_state_key] = tentative_g_score
                    open_set.append((new_state, path + [action], tentative_g_score))

        return None  # No valid path found

    def is_goal(self, state):
        """
        Check if the current state matches the goal state.
        """
        for obj_id in state:
            if state[obj_id] != self.goal_state[obj_id]:
                return False
        return True

    def state_to_tuple(self, state):
        """
        Convert the state dictionary to a tuple for hashing.
        """
        # Convert positions to tuples and sort by object ID
        return tuple(sorted((obj_id, tuple(pos)) for obj_id, pos in state.items()))

# --- Main Execution ---

if __name__ == "__main__":
    # Paths to the JSON files
    current_json_file = '/home/percy/princeton/visual_wholebody/third_party/TAMP/current.json'
    target_json_file = '/home/percy/princeton/visual_wholebody/third_party/TAMP/target.json'

    # Read the JSON files
    current_data = read_json_file(current_json_file)
    target_data = read_json_file(target_json_file)

    # print("current_data, target_data: ", current_data, target_data)
    if current_data is None or target_data is None:
        print("Error reading JSON files.")
        exit(1)

    # Filter objects in current_data to include only IDs 63, 211, 236
    selected_ids = [63, 211, 236]
    filtered_current_data = {}

    for obj_key in current_data:
        obj = current_data[obj_key]
        if obj['id'] in selected_ids:
            filtered_current_data[obj_key] = obj

    # Mapping from current object IDs to target object keys
    id_mapping = {
        63: 'object_001',
        211: 'object_002',
        236: 'object_003'
    }

    initial_state = {}
    goal_state = {}

    for current_obj_key, current_obj in filtered_current_data.items():
        current_id = current_obj['id']
        if current_id in id_mapping:
            target_key = id_mapping[current_id]
            target_obj = target_data.get(target_key)
            if target_obj is None:
                print(f"Target object {target_key} not found in target.json.")
                continue

            # Use the target object's 'id' as the consistent object ID
            obj_id = target_obj['id']

            # Get positions
            initial_position = tuple(current_obj['bbox_center'])
            goal_position = tuple(target_obj['bbox_center'])

            initial_state[obj_id] = initial_position
            goal_state[obj_id] = goal_position
        else:
            print(f"No mapping found for current object ID {current_id}.")

    # Ensure that both states have the same objects
    if set(initial_state.keys()) != set(goal_state.keys()):
        print("Objects in initial and goal states do not match after mapping.")
        print(f"Initial state IDs: {set(initial_state.keys())}")
        print(f"Goal state IDs: {set(goal_state.keys())}")
        exit(1)

    # Create an instance of AStarObjectMover
    mover = AStarObjectMover(initial_state, goal_state)

    # Execute A* search to find the optimal action sequence
    action_sequence = mover.a_star_search()

    if action_sequence:
        print("Optimal sequence of actions:")
        for action in action_sequence:
            action_type, obj_id, target_pos = action
            print(f"{action_type} object {obj_id} to position {target_pos}")
    else:
        print("No valid sequence found")