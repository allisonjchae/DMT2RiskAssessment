"""
Utility functions.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import matplotlib
import numpy as np
import os
import torch
import random


def plot_config():
    """
    Plot configuration variables.
    Input:
        None.
    Returns:
        None.
    """
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams.update({"font.size": 18})


def seed_everything(seed: int, use_deterministic: bool = True) -> None:
    """
    Random state initialization function. Should be called before training.
    Input:
        seed: random seed.
        use_deterministic: whether to only use deterministic algorithms.
    Returns:
        None.
    """
    torch.use_deterministic_algorithms(use_deterministic)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = use_deterministic
