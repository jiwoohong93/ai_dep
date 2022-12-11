import os
import json
import numpy as np
from copy import deepcopy


def save_result(path, stats):
    
    # mean and median
    mean_stats, median_stats = deepcopy(stats), deepcopy(stats)
    for key in stats.keys():
        mean_stats[key] = np.mean(stats[key]).item()
        median_stats[key] = np.median(stats[key]).item()
        
    # save
    with open(os.path.join(path, f"mean_result.json"), "w") as f:
        f.write(json.dumps(mean_stats, indent = 4))
        f.close()
    with open(os.path.join(path, f"median_result.json"), "w") as f:
        f.write(json.dumps(median_stats, indent = 4))
        f.close()