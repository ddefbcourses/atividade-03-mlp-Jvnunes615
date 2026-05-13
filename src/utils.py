import random

import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def normalize_images(X):
    return X / 255.0
