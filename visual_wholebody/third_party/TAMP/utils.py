import json
import numpy as np
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, directed_hausdorff
from frechetdist import frdist
from sklearn.metrics.pairwise import cosine_similarity

def print_nodes(data, path=""):
    """Recursively prints the nodes of a JSON object."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            print_nodes(value, new_path)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = f"{path}[{index}]"
            print_nodes(item, new_path)
    else:
        # This is a leaf node (not a dictionary or list)
        print(f"{path}: {data}")

def read_json_file(filename):
    """Reads a JSON file and returns the data."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {filename}.")
        return None

def find_objects_by_id(data, target_id):
    """Searches for objects with matching IDs and returns them."""
    # Iterate through the top-level objects
    for key, value in data.items():
        if value['id'] == target_id:
            return value
    return {}

def calculate_euclidean_distance(node1, node2):
    """
    Calculates the Euclidean distance between the bbox_center of two nodes.
    """
    # Extract the centers from both nodes
    center1 = node1['bbox_center']
    center2 = node2['bbox_center']

    # Calculate Euclidean distance between centers
    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(center1, center2)))

    return distance

def calculateTrajectorySimilarity(trajectory1, trajectory2, method="dtw", threshold=0.5):
    if np.isnan(trajectory1).any() or np.isnan(trajectory2).any(): return np.infty
    if method == "dtw":
        distance, _ = fastdtw(trajectory1, trajectory2, dist=euclidean)
        return distance
    
    elif method == "lcss":
        m, n = len(trajectory1), len(trajectory2)
        L = np.zeros((m + 1, n + 1))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if np.linalg.norm(trajectory1[i - 1] - trajectory2[j - 1]) < threshold:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        
        return -L[m][n] / min(m, n)
    
    elif method == "haus":
        return max(directed_hausdorff(trajectory1, trajectory2)[0], directed_hausdorff(trajectory2, trajectory1)[0])

    elif method == "frec":
        return frdist(trajectory1, trajectory2)
    
    elif method == "edr":
        m, n = len(trajectory1), len(trajectory2)
        EDR = np.zeros((m + 1, n + 1))
        
        for i in range(1, m + 1):
            EDR[i][0] = i
        for j in range(1, n + 1):
            EDR[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if np.linalg.norm(trajectory1[i - 1] - trajectory2[j - 1]) < threshold:
                    cost = 0
                else:
                    cost = 1
                EDR[i][j] = min(EDR[i - 1][j] + 1, EDR[i][j - 1] + 1, EDR[i - 1][j - 1] + cost)
        
        return EDR[m][n] / max(m, n)
    
    elif method == "euc":
        return np.linalg.norm(trajectory1 - trajectory2)
    
    elif method == "cos":
        return -cosine_similarity([trajectory1.flatten()], [trajectory2.flatten()])[0][0]
     
    elif method == "man":
        return np.sum(np.abs(trajectory1 - trajectory2))
    
    else:
        raise NotImplementedError(f"The method '{method}' is not implemented.")

if __name__ == "__main__":
    file1 = '/home/percy/Dev/intel_realsense_ws/src/A* search/current.json'
    file2 = '/home/percy/Dev/intel_realsense_ws/src/A* search/target.json'

    data1 = read_json_file(file1)
    data2 = read_json_file(file2)

    if data1:
        print(f"Nodes in {file1}:")
        print_nodes(data1)

    if data2:
        print(f"\nNodes in {file2}:")
        print_nodes(data2)

