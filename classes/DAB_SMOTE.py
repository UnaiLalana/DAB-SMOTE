import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

class DAB_SMOTE:
    def __init__(self, r = 1.5, distMethod = "euclidean", k = 1, max_tries_until_change = 10, max_iter = 10000, random_state = 42):
        self.__r__ = r
        self.__distMethod__ = distMethod
        self.__k__ = k
        self.__max_tries_until_change__ = max_tries_until_change
        self.__max_iter__ = max_iter
        self.__n_removed__ = -1
        self.__random_state__ = random_state
        self.__status_code__ = 0

    def __euclideanDist__(self, xi, xmean):
        dist = np.sqrt(np.sum((xi - xmean)**2))
        return dist
    
    def __manhattanDist__(self, xi, xmean):
        dist = np.sum(np.abs(xi - xmean))
        return dist
    
    def __chebyshevDist__(self, xi, xmean):
        dist = np.max(xi-xmean)
        return dist
    
    def __removeNoisySamples__(self, Xmin):
        Xmin_mean = np.mean(Xmin, axis=0)
        
        distMethods = {"euclidean": self.__euclideanDist__, "manhattan": self.__manhattanDist__, "chebyshev": self.__chebyshevDist__}
        
        dists = np.zeros(np.shape(Xmin)[0])
        
        N = np.shape(Xmin)[0]
        
        for i in range(len(Xmin)):
            dists[i] = distMethods[self.__distMethod__](Xmin[i], Xmin_mean)
        
        dists_sort = np.sort(dists)
        
        Q1 = dists_sort[int(np.round((N+1)*0.25))]
        Q3 = dists_sort[int(np.round((N+1)*0.75))]
        IQR = Q3 - Q1
        ub = Q1 + self.__r__ * IQR
        
        delete = []
        
        for i in range(len(dists)):
            if dists[i] > ub:
                delete.append(i)
        
        self.__n_removed__ = len(delete)
        Xmin = np.delete(Xmin, delete, axis = 0)
                
        return Xmin
    
    
    def __screenBoundarySamples__(self, Xmin, clusters):
        k = self.__k__
            
        etiqueta_cluster = np.unique(clusters)
    
        boundaries_totales = []
        for x in etiqueta_cluster:
            XminCl = Xmin[clusters == x]
            ajs = np.mean(XminCl, axis=0)
            ojs = np.std(XminCl, axis=0) 
            boundaries = []
            for j in range(XminCl.shape[1]):
                for i in range(XminCl.shape[0]):
                    if np.abs(XminCl[i, j] - ajs[j]) > (ojs[j] * k):
                        boundaries.append(XminCl[i])
            boundaries_totales.append(boundaries)
        
        return boundaries_totales
    
    
    def __clustering__(self, Xmin):
        db = DBSCAN(eps=0.2, min_samples=10).fit(Xmin)
        clusters = db.labels_


        noise_indices = np.where(clusters == -1)[0]
        cluster_indices = np.where(clusters != -1)[0]

        if len(noise_indices) > 0:
            if len(cluster_indices) > 0:

                closest_clusters, _ = pairwise_distances_argmin_min(Xmin[noise_indices], Xmin[cluster_indices])

                for noise_idx, closest_idx in zip(noise_indices, closest_clusters):
                    clusters[noise_idx] = clusters[cluster_indices[closest_idx]]

            else:
                clusters[noise_indices] = 0
                
        
        unique_clusters = sorted(set(clusters) - {-1})
        centers_new = []
    
        for cluster in unique_clusters:
            cluster_points = Xmin[clusters == cluster]
            center = cluster_points.mean(axis=0) 
            centers_new.append(center)

        centers_new = np.array(centers_new)
        
        return centers_new, clusters
    
    
    def __generateNewSamples__(self,Xmin, boundaries, clusters, centers, N):
        new_samples = []
        etiqueta_cluster = np.unique(clusters)
        for x in etiqueta_cluster:
            XminCl = Xmin[clusters == x]
            boundariesCl = np.array(boundaries[x])
            n_new_samples_Cl = int(np.round(XminCl.shape[0]*N/Xmin.shape[0]))
            cl = centers[x]
            if boundariesCl.shape[0] != 0:
                for n in range(n_new_samples_Cl):
                    xl_index = np.random.randint(boundariesCl.shape[0])
                    xl = boundariesCl[xl_index]

                    yl_index = np.random.randint(XminCl.shape[0])
                    yl = XminCl[yl_index]

                    tries_until_change = 0
                    total_tries = 0
                    while np.array_equal(yl,xl) or np.any(np.all(yl == centers, axis=1)) or self.__euclideanDist__(xl, yl) > self.__euclideanDist__(xl, cl):
                        total_tries += 1
                        tries_until_change += 1
                        yl_index = np.random.randint(XminCl.shape[0])
                        yl = XminCl[yl_index]

                        if tries_until_change > self.__max_tries_until_change__:
                            tries_until_change = 0
                            xl_index = np.random.randint(boundariesCl.shape[0])
                            xl = boundariesCl[xl_index]

                            yl_index = np.random.randint(XminCl.shape[0])
                            yl = XminCl[yl_index]
                        if total_tries > self.__max_iter__:
                            return None

                    t1 = xl + np.random.rand() * ( yl - xl)
                    s1 = t1 + np.random.rand() * (cl - t1)
                    new_samples.append(s1)
        return np.array(new_samples)
    
    def fit_resample(self, X, y):
        self.__X__ = X
        self.__y__ = y
        np.random.seed(self.__random_state__)
        labels, counts = np.unique(self.__y__, return_counts=True)
        minLabel = labels[np.argmin(counts)]
        ymin = self.__y__[self.__y__ == minLabel]
        Xmin = self.__X__[self.__y__ == minLabel]
        
        N = np.max(counts) - np.min(counts)
        
        Xmin_removed = self.__removeNoisySamples__(Xmin)

        
        centers, clusters = self.__clustering__(Xmin_removed)

        
        boundaries = self.__screenBoundarySamples__(Xmin_removed, clusters)
        
        new_samples = self.__generateNewSamples__(Xmin_removed, boundaries, clusters, centers, N)
        
        if new_samples is None or new_samples.shape[0] == 0:
            self.__status_code__ = 2
            return self.__X__, self.__y__
        
        Xnew = np.vstack((self.__X__, new_samples))
        ynew = np.hstack((self.__y__, np.array([minLabel]*new_samples.shape[0])))
        self.__status_code__ = 1
        return Xnew, ynew
    
    def n_examples_deleted(self):
        return self.__n_removed__
    
    def status(self):
        return self.__status_code__, {0: "Resample function not called.", 1: "Resample Succeeded.", 2: "Resample Failed, returning original Data."}.get(self.__status_code__)