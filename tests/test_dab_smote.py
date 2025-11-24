import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.DAB_SMOTE import DAB_SMOTE

import numpy as np

def test_fit_resample_balances_classes():
    """
    Test if fit_resample balances the classes and increases the number of samples.

    This test verifies that:
    1. The number of samples in the minority class equals the majority class after resampling.
    2. The total number of samples increases.
    """
    X = np.random.rand(100, 2)
    y = np.array([0]*90 + [1]*10)

    dab = DAB_SMOTE(random_state=0)
    X_res, y_res = dab.fit_resample(X, y)

    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1], "Classes should be balanced after resampling"
    assert X_res.shape[0] > X.shape[0], "Number of samples should increase after resampling"

def test_remove_noisy_samples_removes_outliers():
    """
    Test if _remove_noisy_samples correctly identifies and removes outliers.

    This test creates a dataset with a clear outlier and asserts that the
    filtered dataset has fewer samples than the original one.
    """
    X = np.vstack([np.random.normal(0, 1, (20, 2)), np.array([[10, 10]])])
    dab = DAB_SMOTE()

    X_filtered = dab._remove_noisy_samples(X)
    assert X_filtered.shape[0] < X.shape[0], "Noisy sample should be removed"

def test_clustering_returns_valid_clusters():
    """
    Test if _clustering returns valid cluster centers and assignments.

    This test checks that:
    1. The cluster centers have the correct dimensions.
    2. Every sample is assigned to a valid cluster index (>= 0).
    """
    X = np.random.rand(50, 2)
    dab = DAB_SMOTE()

    centers, clusters = dab._clustering(X)

    assert centers.ndim == 2
    assert len(clusters) == len(X)
    assert np.all(clusters >= 0), "All samples must be assigned to a cluster"


def test_generate_new_samples_creates_points():
    """
    Test if _generate_new_samples creates new synthetic samples.

    This test verifies that:
    1. The function returns a non-None result.
    2. The returned array has samples (rows > 0).
    3. The number of features remains consistent with the input.
    """
    X = np.random.rand(20, 2)
    dab = DAB_SMOTE()
    centers, clusters = dab._clustering(X)
    boundaries = dab._screen_boundary_samples(X, clusters)

    new_samples = dab._generate_new_samples(X, boundaries, clusters, centers, N=10)
    assert new_samples is not None
    assert new_samples.shape[0] > 0
    assert new_samples.shape[1] == X.shape[1]

def test_summary_reflects_process_status():
    """
    Test if the summary property contains expected keys and values.

    This test ensures that after fitting, the summary dictionary includes:
    1. A "Status code".
    2. Non-negative "Number of clusters".
    3. Non-negative "Number of examples generated".
    """
    X = np.random.rand(100, 2)
    y = np.array([0]*80 + [1]*20)
    dab = DAB_SMOTE()
    dab.fit_resample(X, y)
    summary = dab.summary

    assert "Status code" in summary
    assert summary["Number of clusters"] >= 0
    assert summary["Number of examples generated"] >= 0

def test_clustering_density_solver():
    """
    Test that _clustering works correctly when using the 'density' solver.

    Ensures that:
    1. Density-based centers are computed.
    2. The output dimensions are valid.
    """
    X = np.random.rand(50, 2)
    dab = DAB_SMOTE(solver="density")

    centers, clusters = dab._clustering(X)

    assert centers.ndim == 2
    assert centers.shape[1] == X.shape[1]
    assert len(clusters) == len(X)

def test_fit_resample_returns_original_when_generation_fails_safe():
    """
    Force _generate_new_samples to fail (return None) so fit_resample
    returns the original X and y.

    Uses enough minority samples to avoid IndexError in _remove_noisy_samples.
    """
    X = np.random.rand(30, 2)
    y = np.array([0]*25 + [1]*5)

    dab = DAB_SMOTE()

    dab._generate_new_samples = lambda *args, **kwargs: None

    X_res, y_res = dab.fit_resample(X, y)

    assert np.allclose(X_res, X)
    assert np.array_equal(y_res, y)

def test_clustering_all_noise_assignment():
    """
    Test _clustering correctly assigns noise points to closest clusters.
    """
    cluster1 = np.random.rand(10, 2) + 0
    cluster2 = np.random.rand(10, 2) + 5
    noise = np.array([[20, 20], [25, 25]])

    X = np.vstack([cluster1, cluster2, noise])
    dab = DAB_SMOTE()
    
    centers, clusters = dab._clustering(X)

    assert np.all(clusters >= 0)
    
    assert clusters[-2] in clusters[:-2]
    assert clusters[-1] in clusters[:-2]

def test_generate_new_samples_returns_none_when_max_iter_exceeded():
    """
    Test _generate_new_samples returns None if max_iter is exceeded.
    """
    X = np.random.rand(5, 2)
    clusters = np.zeros(5, dtype=int)
    centers = np.mean(X, axis=0).reshape(1, -1)
    boundaries = [X.copy()]

    dab = DAB_SMOTE(max_iter=1)
    
    result = dab._generate_new_samples(X, boundaries, clusters, centers, N=10)

    assert result is None