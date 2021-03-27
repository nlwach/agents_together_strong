import os

import numpy as np


def get_analysis_data(agent, attr=None):
    attributes = f"_{attr}" if attr is not None else ""
    file_name = f"scores{attributes}.npy"
    score_file = os.path.join("agent_code", agent, file_name)
    if not os.path.isfile(score_file):
        raise ValueError(f"File {score_file} not found!")
    return np.load(score_file)
