###############################
#
#  Helpful utilities for processing actions, observations.
#
###############################

import numpy as np
import jax.numpy as jnp

class ActionTransform():
    pass

class ActionDiscretizeBins(ActionTransform):
    def __init__(self, bins_per_dim, action_dim):
        self.bins_per_dim = bins_per_dim
        self.action_dim = action_dim
        self.bins = np.linspace(-1, 1, bins_per_dim + 1)

    # Assumes action is in [-1, 1].
    def action_to_ids(self, action):
        ids = np.digitize(action, self.bins) - 1
        ids = np.clip(ids, 0, self.bins_per_dim - 1)
        return ids

    def ids_to_action(self, ids):
        action = (self.bins[ids] + self.bins[ids + 1]) / 2
        return action
    
class ActionDiscretizeCluster(ActionTransform):
    def __init__(self, num_clusters, data_actions):
        self.num_clusters = num_clusters
        assert len(data_actions.shape) == 2 # (data_size, action_dim)
        print("Clustering actions of shape", data_actions.shape)

        # Cluster the data.
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data_actions)
        self.centers = kmeans.cluster_centers_
        # self.labels = kmeans.labels_
        self.centers = jnp.array(self.centers)
        print("Average cluster error is", kmeans.inertia_ / len(data_actions))
        print("Average cluster error per dimension is", (kmeans.inertia_ / len(data_actions)) / data_actions.shape[1])
        # print(self.centers.shape)

    def action_to_ids(self, action):
        if len(action.shape) == 1:
            action = action[None]
        assert len(action.shape) == 2 # (batch, action_dim,)
        # Find the closest cluster center.
        dists = jnp.linalg.norm(self.centers[None] - action[:, None], axis=-1)
        ids = jnp.argmin(dists, axis=-1)
        return ids
    
    def ids_to_action(self, ids):
        action = self.centers[ids]
        return action
    
# Test
# action_discretize_bins = ActionDiscretizeBins(32, 2)
# action = np.array([-1, -0.999, -0.5, 0, 0.5, 0.999, 1])
# ids = action_discretize_bins.action_to_ids(action)
# print(ids)
# action_recreate = action_discretize_bins.ids_to_action(ids)
# print(action_recreate)
# assert np.abs(action - action_recreate).max() < 0.1

# action_discretize_cluster = ActionDiscretizeCluster(32, np.random.uniform(low=-1, high=1, size=(10000, 1)))
# action = np.array([-1, -0.999, -0.5, 0, 0.5, 0.999, 1])[:, None] # [7, 1]
# ids = action_discretize_cluster.action_to_ids(action)
# print(ids)
# action_recreate = action_discretize_cluster.ids_to_action(ids)
# print(action_recreate)
# assert np.abs(action - action_recreate).max() < 0.1