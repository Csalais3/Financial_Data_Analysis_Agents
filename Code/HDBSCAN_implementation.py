import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.merge_lambda = np.zeros(n)
    
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
            
        return self.parent[a]
    
    def union(self, a, b, lam):
        root_a = self.find(a)
        root_b = self.find(b)
        
        if root_a == root_b:
            return False

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
            
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.merge_lambda[root_a] = lam
        
        return True
            

class HDBSCAN:
    def __init__(self, X, min_samples, threshold_percentile):
        self.X = X
        self.min_samples = min_samples
        self.threshold_percentile = threshold_percentile
    
    def calculate_core_distances(self, X, min_samples):
        pass
    
    def calculate_mutual_reachability(self, core_dists, pairwise_dists):
        pass
    
    def build_mst(self, mrd):
        pass
    
    def buld_cluster_heirarchy(self, n_points, edges):
        pass
    
    def extract_clusters(self, ):
        pass
        
    def hdbscan(self, X, min_samples, threshold_percentile):
        pass
         