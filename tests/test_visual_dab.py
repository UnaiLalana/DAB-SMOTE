import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.Visual_DAB import Visual_DAB
import numpy as np

def test_visual_dab_methods_work():
    """
    Test if Visual_DAB methods execute without errors and return expected structures.

    This test runs the pipeline of methods:
    1. get_removed_samples
    2. get_clustering
    3. get_screened_boundaries
    4. get_generated_samples

    And asserts that the outputs are of the correct types (ndarrays or lists).
    """
    X = np.random.rand(30, 2)
    visual = Visual_DAB()

    X_filtered = visual.get_removed_samples(X)
    centers, clusters = visual.get_clustering(X_filtered)
    boundaries = visual.get_screened_boundaries(X_filtered, clusters)
    new_samples = visual.get_generated_samples(X_filtered, boundaries, clusters, centers, N=5)

    assert isinstance(X_filtered, np.ndarray)
    assert isinstance(clusters, np.ndarray)
    assert isinstance(boundaries, list)
    assert new_samples is None or isinstance(new_samples, np.ndarray)

def test_visual_dab_returns_correct_types():
    """
    Test if Visual_DAB methods return objects of the correct type.

    Specifically checks:
    1. get_removed_samples returns a numpy array.
    2. get_clustering returns two numpy arrays (centers and clusters).
    """
    X = np.random.rand(10, 3)
    visual = Visual_DAB()

    filtered = visual.get_removed_samples(X)
    assert isinstance(filtered, np.ndarray)

    centers, clusters = visual.get_clustering(filtered)
    assert isinstance(centers, np.ndarray)
    assert isinstance(clusters, np.ndarray)
