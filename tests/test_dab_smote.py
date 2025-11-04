import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.DAB_SMOTE import DAB_SMOTE

import numpy as np

def test_fit_resample_balances_classes():
    X = np.random.rand(100, 2)
    y = np.array([0]*90 + [1]*10)

    dab = DAB_SMOTE(random_state=0)
    X_res, y_res = dab.fit_resample(X, y)

    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1], "Classes should be balanced after resampling"
    assert X_res.shape[0] > X.shape[0], "Number of samples should increase after resampling"

def test_remove_noisy_samples_removes_outliers():
    X = np.vstack([np.random.normal(0, 1, (20, 2)), np.array([[10, 10]])])
    dab = DAB_SMOTE()

    X_filtered = dab._removeNoisySamples(X)
    assert X_filtered.shape[0] < X.shape[0], "Noisy sample should be removed"

def test_clustering_returns_valid_clusters():
    X = np.random.rand(50, 2)
    dab = DAB_SMOTE()

    centers, clusters = dab._clustering(X)

    assert centers.ndim == 2
    assert len(clusters) == len(X)
    assert np.all(clusters >= 0), "All samples must be assigned to a cluster"


def test_generate_new_samples_creates_points():
    X = np.random.rand(20, 2)
    dab = DAB_SMOTE()
    centers, clusters = dab._clustering(X)
    boundaries = dab._screenBoundarySamples(X, clusters)

    new_samples = dab._generateNewSamples(X, boundaries, clusters, centers, N=10)
    assert new_samples is not None
    assert new_samples.shape[0] > 0
    assert new_samples.shape[1] == X.shape[1]

def test_summary_reflects_process_status():
    X = np.random.rand(100, 2)
    y = np.array([0]*80 + [1]*20)
    dab = DAB_SMOTE()
    dab.fit_resample(X, y)
    summary = dab.summary

    assert "Status code" in summary
    assert summary["Number of clusters"] >= 0
    assert summary["Number of examples generated"] >= 0


