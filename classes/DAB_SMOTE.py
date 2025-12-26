"""
DAB_SMOTE.py
============

Implementation of DAB-SMOTE (Density Aware Borderline Synthetic Minority Oversampling Technique).

This module provides a data-level resampling approach designed to address class imbalance problems
by generating synthetic samples for the minority class. It combines boundary detection,
noise removal, and density-based clustering to improve the representativeness of synthetic samples.

References
----------
- U. Lalana and J. A. S. Delgado, ‘Estudio, análisis e implementación de FSDR-SMOTE,
técnica de sobremuestreo para problemas de clasificación desbalanceados’,
Universidad Publica de Navarra.
https://academica-e.unavarra.es/entities/publication/94317fe9-1d28-408e-933b-1c3195a9c983

Author
------
Unai Lalana
"""

from typing import Callable
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class DAB_SMOTE:
    """
    Density and Boundary-based Synthetic Minority Oversampling Technique (DAB-SMOTE).

    This method generates new synthetic samples for the minority class(es). It supports both
    binary and multiclass classification problems and automatically detects the type of problem
    based on the input data.

    The process involves:
    1. Identifying minority class(es).
    2. Removing noisy samples using IQR-based filtering.
    3. Clustering the remaining data with DBSCAN.
    4. Detecting boundary samples within clusters.
    5. Interpolating between boundary, cluster, and central points to create synthetic samples.

    Parameters
    ----------
    r : float, default=1.5
        Multiplier for IQR when filtering noisy samples.
    dist_method : {'euclidean', 'manhattan', 'chebyshev'}, default='euclidean'
        Distance metric used for noise filtering.
    k : int, default=1
        Standard deviation multiplier for boundary sample detection.
    eps : float, default=0.75
        Epsilon parameter for DBSCAN clustering.
    min_samples : int, default=10
        Minimum number of samples required to form a cluster.
    sampling_strategy :
    float | str | dict | list | Callable[[np.ndarray, np.ndarray], dict], default='auto'
        Sampling strategy to use for generating synthetic samples.
    max_tries_until_change : int, default=10
        Maximum number of retries before changing boundary samples.
    max_iter : int, default=10000
        Maximum number of total iterations allowed during sample generation.
    random_state : int, default=42
        Random seed for reproducibility.
    solver : {'means', 'density'}, default='means'
        Method used to calculate cluster centers ('means' or 'density').
    progress : bool, default=False
        If True, shows a progress bar during sample generation.

    Attributes
    ----------
    n_removed_ : int
        Number of noisy samples removed.
    number_of_clusters_ : int
        Number of clusters found by DBSCAN.
    number_of_examples_generated_ : int
        Number of new synthetic examples generated.
    border_samples_percent_ : float
        Ratio of boundary samples to total samples.
    status_code_ : int
        Status code for the resampling process:
        - 0: Not executed
        - 1: Success
        - 2: Failed (returns original data)
    """

    def __init__(
        self,
        r: float = 1.5,
        dist_method: str = "euclidean",
        k: float = 1,
        eps: float = 0.75,
        min_samples: int = 10,
        sampling_strategy: (
            float | str | dict | list | Callable[[np.ndarray, np.ndarray], dict]
        ) = "auto",
        max_tries_until_change: int = 10,
        max_iter: int = 10000,
        random_state: int = 42,
        solver: str = "means",
        progress: bool = False,
    ) -> None:
        self._r = r
        self._dist_method = dist_method
        self._k = k
        self._dist_methods = {
            "euclidean": lambda a, b: np.sqrt(np.sum((a - b) ** 2)),
            "manhattan": lambda a, b: np.sum(np.abs(a - b)),
            "chebyshev": lambda a, b: np.max(np.abs(a - b)),
        }
        self._eps = eps
        self._min_samples = min_samples
        self._sampling_strategy = sampling_strategy
        self._max_tries_until_change = max_tries_until_change
        self._max_iter = max_iter
        self._solver = solver
        self._n_removed = -1
        self._random_state = random_state
        self._progress = progress
        self._number_of_clusters = 0
        self._multiclass = False
        self._number_of_examples_generated = 0
        self._border_samples_percent = 0
        self._status_code = 0

    def _remove_noisy_samples(self, X_min: np.ndarray) -> np.ndarray:
        """
        Remove noisy samples based on the interquartile range (IQR) method.

        Parameters
        ----------
        X_min : ndarray of shape (n_samples, n_features)
            Minority class samples.

        Returns
        -------
        X_min : ndarray
            Minority samples with noisy points removed.
        """
        X_min_mean = np.mean(X_min, axis=0)
        dists = np.zeros(np.shape(X_min)[0])
        N = np.shape(X_min)[0]

        for i, x in enumerate(X_min):
            dists[i] = self._dist_methods[self._dist_method](x, X_min_mean)

        dists_sort = np.sort(dists)
        Q1 = dists_sort[int(np.round((N + 1) * 0.25))]
        Q3 = dists_sort[int(np.round((N + 1) * 0.75))]
        IQR = Q3 - Q1
        ub = Q1 + self._r * IQR

        delete = []
        for i, dist in enumerate(dists):
            if dist > ub:
                delete.append(i)

        self._n_removed = len(delete)
        X_min = np.delete(X_min, delete, axis=0)
        return X_min

    def _screen_boundary_samples(self, X_min: np.ndarray, clusters: np.ndarray) -> list:
        """
        Identify boundary samples in each cluster.

        Parameters
        ----------
        X_min : ndarray
            Minority class samples.
        clusters : ndarray
            Cluster labels assigned by DBSCAN.

        Returns
        -------
        list of list
            Boundary samples detected per cluster.
        """
        k = self._k
        cluster_label = np.unique(clusters)
        all_boundaries = []

        for x in cluster_label:
            X_min_cl = X_min[clusters == x]
            ajs = np.mean(X_min_cl, axis=0)
            ojs = np.std(X_min_cl, axis=0)
            boundaries = []
            for j in range(X_min_cl.shape[1]):
                for i in range(X_min_cl.shape[0]):
                    if np.abs(X_min_cl[i, j] - ajs[j]) > (ojs[j] * k):
                        boundaries.append(X_min_cl[i])
            all_boundaries.append(boundaries)

        self._border_samples_percent = len(all_boundaries) / X_min.shape[0]
        return all_boundaries

    def _clustering(self, X_min: np.ndarray) -> tuple:
        """
        Cluster minority samples using DBSCAN and compute cluster centers.

        Parameters
        ----------
        X_min : ndarray
            Minority class samples.

        Returns
        -------
        centers_new : ndarray
            Computed cluster centers.
        clusters : ndarray
            Cluster labels assigned to each sample.
        """
        db = DBSCAN(eps=self._eps, min_samples=self._min_samples).fit(X_min)
        clusters = db.labels_

        noise_indices = np.where(clusters == -1)[0]
        cluster_indices = np.where(clusters != -1)[0]

        if len(noise_indices) > 0:
            if len(cluster_indices) > 0:
                closest_clusters, _ = pairwise_distances_argmin_min(
                    X_min[noise_indices], X_min[cluster_indices]
                )
                for noise_idx, closest_idx in zip(noise_indices, closest_clusters):
                    clusters[noise_idx] = clusters[cluster_indices[closest_idx]]
            else:
                clusters[noise_indices] = 0

        unique_clusters = sorted(set(clusters) - {-1})
        self._number_of_clusters = len(unique_clusters)
        centers_new = []

        if self._solver == "means":
            for cluster in unique_clusters:
                cluster_points = X_min[clusters == cluster]
                center = cluster_points.mean(axis=0)
                centers_new.append(center)
        elif self._solver == "density":
            for cluster in unique_clusters:
                cluster_points = X_min[clusters == cluster]
                nbrs = NearestNeighbors(radius=self._eps).fit(cluster_points)
                radii_neighbors = nbrs.radius_neighbors(
                    cluster_points, return_distance=False
                )
                neighbor_counts = np.array([len(neigh) for neigh in radii_neighbors])
                most_dense_index = np.argmax(neighbor_counts)
                most_dense_point = cluster_points[most_dense_index]
                centers_new.append(most_dense_point)

        centers_new = np.array(centers_new)
        return centers_new, clusters

    def _generate_new_samples(
        self,
        X_min: np.ndarray,
        boundaries: list,
        clusters: np.ndarray,
        centers: np.ndarray,
        N: int,
    ) -> np.ndarray | None:
        """
        Generate new synthetic samples from cluster boundaries and centers.

        Parameters
        ----------
        X_min : ndarray
            Minority samples after cleaning.
        boundaries : list of list
            Detected boundary samples.
        clusters : ndarray
            Cluster labels.
        centers : ndarray
            Cluster centers.
        N : int
            Number of synthetic samples to generate.

        Returns
        -------
        ndarray
            Array of new synthetic samples.
        """
        new_samples = []
        cluster_label = np.unique(clusters)
        cluster_map = {
            x: (X_min[clusters == x], np.array(boundaries[x]), centers[x])
            for x in cluster_label
        }
        cluster_cycle = []

        cluster_sizes = [X_min[clusters == x].shape[0] for x in cluster_label]
        total_size = sum(cluster_sizes)

        raw_quota = [(size * N) / total_size for size in cluster_sizes]

        base_quota = [int(np.floor(q)) for q in raw_quota]

        residuals = [q - b for q, b in zip(raw_quota, base_quota)]

        missing = N - sum(base_quota)

        if missing > 0:
            idxs = np.argsort(residuals)[::-1]
            for i in idxs[:missing]:
                base_quota[i] += 1

        for x, quota in zip(cluster_label, base_quota):
            X_min_cl, boundaries_cl, cl = cluster_map[x]
            cluster_cycle.extend([(x, boundaries_cl, X_min_cl, cl)] * quota)

        np.random.shuffle(cluster_cycle)
        iterable = (
            tqdm(cluster_cycle, total=len(cluster_cycle))
            if self._progress
            else cluster_cycle
        )

        for x, boundaries_cl, X_min_cl, cl in iterable:
            xl_index = np.random.randint(boundaries_cl.shape[0])
            xl = boundaries_cl[xl_index]
            yl_index = np.random.randint(X_min_cl.shape[0])
            yl = X_min_cl[yl_index]

            tries_until_change = 0
            total_tries = 0
            while (
                np.array_equal(yl, xl)
                or np.any(np.all(yl == centers, axis=1))
                or self._dist_methods["euclidean"](xl, yl)
                > self._dist_methods["euclidean"](xl, cl)
            ):
                total_tries += 1
                tries_until_change += 1
                yl_index = np.random.randint(X_min_cl.shape[0])
                yl = X_min_cl[yl_index]

                if tries_until_change > self._max_tries_until_change:
                    tries_until_change = 0
                    xl_index = np.random.randint(boundaries_cl.shape[0])
                    xl = boundaries_cl[xl_index]
                    yl_index = np.random.randint(X_min_cl.shape[0])
                    yl = X_min_cl[yl_index]

                if total_tries > self._max_iter:
                    return None

            t1 = xl + np.random.rand() * (yl - xl)
            s1 = t1 + np.random.rand() * (cl - t1)
            new_samples.append(s1)

        self._number_of_examples_generated = len(new_samples)
        return np.array(new_samples)

    def _define_sampling_strategy(
        self,
        y: np.ndarray,
        sampling_strategy: (
            dict | str | float | list | Callable[[np.ndarray, np.ndarray], dict]
        ) = "auto",
    ) -> dict:
        """
        define the sampling strategy based on the input parameters.

        Parameters
        ----------
        y : ndarray
            Class labels.
        sampling_strategy : dict | str | float | list | Callable[[np.ndarray, np.ndarray], dict]
            Sampling strategy.

        Returns
        -------
        dict
            Dictionary of minority class labels and their corresponding counts.
        """
        labels, counts = np.unique(y, return_counts=True)
        major_classes = labels[counts == np.max(counts)]

        if sampling_strategy in ("auto", "not majority"):
            minority_labels = [
                i for i, lbl in enumerate(labels) if lbl not in major_classes
            ]
            max_count = np.max(counts)
            minority_counts = max_count - counts[minority_labels]

            self._multiclass = len(minority_labels) > 1
            return {
                labels[x]: y
                for i, (x, y) in enumerate((zip(minority_labels, minority_counts)))
            }

        if sampling_strategy == "minority":
            return {labels[np.argmin(counts)]: np.max(counts) - np.min(counts)}

        if isinstance(sampling_strategy, dict):
            return sampling_strategy

        if isinstance(sampling_strategy, float):
            if sampling_strategy < 0 or sampling_strategy > 1:
                raise ValueError("Sampling strategy must be between 0 and 1.")
            if len(labels) > 2:
                raise ValueError(
                    "Sampling strategy is not supported for multiclass problems."
                )
            return {
                labels[np.argmin(counts)]: int(
                    np.max(counts) * sampling_strategy - np.min(counts)
                )
            }

        if isinstance(sampling_strategy, list):
            minority_labels = np.setdiff1d(sampling_strategy, major_classes)
            max_count = np.max(counts)
            minority_counts = max_count - counts[minority_labels]

            self._multiclass = len(minority_labels) > 1
            return {
                labels[x]: y
                for i, (x, y) in enumerate((zip(minority_labels, minority_counts)))
            }

        if callable(sampling_strategy):
            return sampling_strategy(y)

        raise ValueError("Invalid sampling strategy.")

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Fit and resample the dataset by generating synthetic samples for the minority class(es).

        This method automatically detects whether the problem is binary or multiclass. In the case
        of multiclass problems, it iterates over each minority class to generate synthetic samples
        until they reach the count of the majority class.

        Parameters
        ----------
        X : ndarray
            Input feature matrix.
        y : ndarray
            Class labels.

        Returns
        -------
        X_new : ndarray
            Resampled feature matrix.
        y_new : ndarray
            Resampled class labels.
        """
        np.random.seed(self._random_state)

        sample = self._define_sampling_strategy(
            y, sampling_strategy=self._sampling_strategy
        )
        new_samples = []

        for lbl, diff in sample.items():
            X_min = X[y == lbl]
            N = diff
            X_min_removed = self._remove_noisy_samples(X_min)
            centers, clusters = self._clustering(X_min_removed)
            boundaries = self._screen_boundary_samples(X_min_removed, clusters)

            new_samples.append(
                self._generate_new_samples(
                    X_min_removed, boundaries, clusters, centers, int(N)
                )
            )

        new_samples = np.vstack(new_samples)
        if new_samples[0][0] is None or new_samples.shape[0] == 0:
            self._status_code = 2
            return X, y
        X_new = np.vstack((X, new_samples))

        y_new = np.hstack((y, np.repeat(list(sample.keys()), list(sample.values()))))
        self._status_code = 1
        return X_new, y_new

    @property
    def summary(self) -> dict:
        """
        Print and return a summary of the resampling process.

        The summary includes information about whether the problem was detected as
        multiclass, the status of the process, and other key metrics.

        Returns
        -------
        dict
            Summary including key performance indicators and problem type detection.
        """
        status_msg = {
            0: "Resample function not called.",
            1: "Resample Succeeded.",
            2: "Resample Failed, returning original Data.",
        }.get(self._status_code)

        summary = {
            "Multiclass": self._multiclass,
            "Status code": self._status_code,
            "Status message": status_msg,
            "Number of examples removed": self._n_removed,
            "Number of clusters": self._number_of_clusters,
            "Number of examples generated": self._number_of_examples_generated,
            "Border samples percentage": self._border_samples_percent,
        }

        print("\n--- Summary ---")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("---------------")
        return summary
