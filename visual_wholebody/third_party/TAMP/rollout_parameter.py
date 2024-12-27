class rolloutPara:
    purpose = 0 # 0: action rollout    1: uncertainty reducing rollout
    num_envs = 50
    class actionPara:
        object_id = 0
        grasp_id = 0
        
    class uncertaintyPara:
        name = ['mass', "shape"]
        mass_range = [[0.2, 1], [2.2, 5.0], [0.8, 3.8]]
        shape_range = [-0.01, 0.01]
        
    class reducUncertainPara:
        object_id = 0
        uncertainty_name = ['mass']