import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.Visual_DAB import Visual_DAB
from classes.DAB_SMOTE import DAB_SMOTE
import numpy as np

def test_visual_dab_methods_work():
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
    X = np.random.rand(10, 3)
    visual = Visual_DAB()

    filtered = visual.get_removed_samples(X)
    assert isinstance(filtered, np.ndarray)

    centers, clusters = visual.get_clustering(filtered)
    assert isinstance(centers, np.ndarray)
    assert isinstance(clusters, np.ndarray)

def test_reproducibility_with_random_state():
    X = np.random.rand(50, 2)
    y = np.array([0]*40 + [1]*10)

    dab1 = DAB_SMOTE(random_state=42)
    dab2 = DAB_SMOTE(random_state=42)

    X1, y1 = dab1.fit_resample(X, y)
    X2, y2 = dab2.fit_resample(X, y)

    assert np.allclose(X1, X2), "Results should be reproducible with the same random_state"
    assert np.array_equal(y1, y2)
