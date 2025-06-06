"""
EffectBasedActionPrototypes
---------
Unsupervised discretization algorithm for effect based action prototype generation.
"""

from copy import copy, deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from .rgng import RobustGrowingNeuralGas
import matplotlib.pyplot as plt 


class EffectActionPrototypes:
    """
    Facilitates unsupervised discretization algorithm for effect based action prototype generation.

    This class is designed to process (effect, motion) collections to identify
    effect based action prototypes. It utilizes clustering techniques to analyze motion
    samples effects and generate prototypes that represent common patterns within the effect space.
    The clustering method varies based on the dimensionality of the effect dimensions provided,
    using histograms for one-dimensional data
    and K-Means clustering for multi-dimensional effect spaces.

    Attributes:
        motion_samples (pandas.DataFrame): A DataFrame containing motion samples with each
            row representing a sample and each column a dimension of motion.
        motion_dimensions (List[str]): Specifies the columns in `motion_samples` that contain
            the relevant motion dimensions for analysis.
        action_prototypes (numpy.ndarray or None): Stores the generated action prototypes
            after processing. Initially set to None.
        m_samples_labeled (pandas.DataFrame or None): Similar to `motion_samples` but includes
            an additional column for cluster labels. Initially set to None.
        prototypes_per_label (Dict[int, numpy.ndarray] or None): Maps each cluster label to its
            corresponding action prototype(s). Initially set to None.
        cluster_labels (Set[int] or None): Contains the unique labels identifying the clusters
            found in the motion data. Initially set to None.
    """

    def __init__(
        self,
        motion_samples: pd.DataFrame,
        motion_dimensions: list,
        log_level=None, # 'DEBUG' for messages
    ) -> None:
        self.motion_samples = copy(motion_samples)
        self.motion_dimensions = motion_dimensions
        self.action_prototypes = None
        self.m_samples_labeled = None
        self.prototypes_per_label = None
        self.fixed_number_of_prototypes = None

        self.__pre_process = None
        self.__prototype_per_cluster_limit = None

        if log_level == 'DEBUG':
            self.log_level = 'DEBUG'
        else:
            self.log_level = 'INFO'

    def log(self, *args):
        """Always print info-level messages."""
        print(f"[INFO]:", *args)

    def log_debug(self, *args):
        """Only print debug messages if log_level is DEBUG."""
        if self.log_level == 'DEBUG':
            print(f"[DEBUG]:", *args)

    def generate(
        self,
        effect_dimensions: list,
        limit_prototypes_per_cluster=10,
        fixed_number_of_prototypes=None
    ) -> np.ndarray:
        """
        Starts prototype generation and returns prototypes. Depending on the number of
        effect dimensions histogram binning or K-Means clustering is used on the effect
        dimensions of the data to categorize the motion samples.
        """
        self.__prototype_per_cluster_limit = limit_prototypes_per_cluster
        self.fixed_number_of_prototypes = fixed_number_of_prototypes
        # Assert effect_dimensions list
        if len(effect_dimensions) == 1 and len(effect_dimensions[0]) == 1:
            #print('Histogram effect clustering')
            cluster_labels = self.__bin_histogram_samples(effect_dimensions[0])
        else:
            #print('Kmeans effect clustering')
            cluster_labels = self.__kmeans_effect_clustering(effect_dimensions)

        self.__generate_prototypes(effect_dimensions, cluster_labels)
        return self.action_prototypes

    def __generate_prototypes(
        self,
        effect_dimensions: list,
        cluster_labels: dict,
    ) -> None:
        effect_dimensions = effect_dimensions[0]
        # Dynamic prototypes per cluster
        if self.fixed_number_of_prototypes is None:
            self.log_debug("Number of prototypes not fixed prototypes")
            mean_stds = []
            for i in cluster_labels:
                cluster_samples = self.m_samples_labeled[
                    self.m_samples_labeled["cluster_label"] == i
                ]
                mean_stds.append(
                    self.__encode_mean_std([cluster_samples], effect_dimensions)
                )

            mean_stds = np.stack(mean_stds)
            mean_std_all_dims = np.add.reduce(mean_stds, axis=1)
            mean_std_all_dims = mean_std_all_dims / np.max(mean_std_all_dims, axis=(0, 1))

            cv = mean_std_all_dims.T[1] / mean_std_all_dims.T[0]
            cv = np.array([min(x, 0.999999) for x in cv])
            max_prototypes_per_cluster = (
                (1 - cv) * self.__prototype_per_cluster_limit * mean_std_all_dims.T[1]
            )
            max_prototypes_per_cluster = max_prototypes_per_cluster / np.min(
                max_prototypes_per_cluster
            )
            max_prototypes_per_cluster = np.floor(max_prototypes_per_cluster)
            max_prototypes_per_cluster = np.array(
                [
                    min(x, self.__prototype_per_cluster_limit)
                    for x in max_prototypes_per_cluster
                ]
            )
        else:
            max_prototypes_per_cluster = [self.fixed_number_of_prototypes] * len(cluster_labels)
        
        self.prototypes_per_label = {}
        self.__pre_process = StandardScaler()
        scaled_m_samples = deepcopy(self.m_samples_labeled)
        scaled_m_samples[self.motion_dimensions] = self.__pre_process.fit_transform(
            scaled_m_samples[self.motion_dimensions]
        )

        for cluster_label, num_prototypes in zip(
            cluster_labels, max_prototypes_per_cluster
        ):
            if num_prototypes == 1:
                self.log_debug("Label:", cluster_label, '#prototypes:', num_prototypes)
                self.__single_prototype_per_class(cluster_label, effect_dimensions)
            else:
                self.log_debug("Label:", cluster_label, '#prototypes:', num_prototypes)
                self.__multi_prototypes(
                    num_prototypes,
                    scaled_m_samples[
                        scaled_m_samples["cluster_label"] == cluster_label
                    ],
                    cluster_label,
                )

    def __single_prototype_per_class(
        self, cluster_label: dict, effect_dimensions: list
    ) -> None:
        single_cluster_df = self.m_samples_labeled[
            self.m_samples_labeled["cluster_label"] == cluster_label
        ]
        cluster_effect_means = single_cluster_df[effect_dimensions].mean().to_numpy()

        closest_motion_to_effect_mean = (
            abs(single_cluster_df[effect_dimensions] - cluster_effect_means)
            .sum(axis=1)
            .argmin()
        )

        prototype = single_cluster_df[self.motion_dimensions].iloc[
            closest_motion_to_effect_mean
        ]

        if self.action_prototypes is None:
            self.action_prototypes = prototype
        else:
            self.action_prototypes = np.vstack((self.action_prototypes, prototype))
        self.prototypes_per_label[cluster_label] = prototype.to_numpy()

    def __multi_prototypes(
        self,
        num_prototypes: int,
        cluster_data: pd.DataFrame,
        cluster_label: int,
    ) -> None:
        # RGNG
        data_np = cluster_data[self.motion_dimensions].to_numpy()
        rgng = RobustGrowingNeuralGas(
            input_data=data_np, max_number_of_nodes=num_prototypes, real_num_clusters=1
        )
        resulting_centers = rgng.fit_network(a_max=100, passes=20) #100 20
        local_prototype = self.__pre_process.inverse_transform(resulting_centers)
        if self.action_prototypes is None:
            self.action_prototypes = local_prototype
            self.prototypes_per_label[cluster_label] = local_prototype
        else:
            self.action_prototypes = np.vstack(
                (self.action_prototypes, local_prototype)
            )
            self.prototypes_per_label[cluster_label] = local_prototype

    def __bin_histogram_samples(self, effect_dimensions: list) -> None:
        hist, bin_edges = np.histogram(self.motion_samples[effect_dimensions[0]])

        label = 0
        cluster_labels_with_edges = {}
        cluster_labels = []
        for i, count in enumerate(hist):
            if count != 0:
                cluster_labels_with_edges[label] = (bin_edges[i], bin_edges[i + 1])
                cluster_labels.append(label)
                label += 1

        self.m_samples_labeled = copy(self.motion_samples)
        self.m_samples_labeled["cluster_label"] = self.m_samples_labeled[
            effect_dimensions[0]
        ].apply(lambda x: self.__find_position_hist(x, cluster_labels_with_edges))

        return set(cluster_labels)

    def __kmeans_effect_clustering(self, effect_dimensions: list) -> None:
        flatten_effect_dims = []
        for e in effect_dimensions:
            flatten_effect_dims = flatten_effect_dims + e

        kmeans_input = np.array(self.motion_samples[flatten_effect_dims])
        norm_factor = np.max(abs(kmeans_input), axis=0)
        kmeans_input = kmeans_input / norm_factor

        range_n_clusters = [3, 4, 5, 6, 7, 8, 9, 10]
        best_score = np.inf
        best_num_of_clusters = 0
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(
                kmeans_input
            )
            dbi = davies_bouldin_score(kmeans_input, kmeans.labels_)
            self.log_debug("Clusters", n_clusters, "score", dbi)
            if dbi < best_score:
                best_score = dbi
                best_num_of_clusters = n_clusters
        kmeans = KMeans(n_clusters=best_num_of_clusters, random_state=0, n_init=10).fit(
            kmeans_input
        )

        self.m_samples_labeled = self.motion_samples
        self.m_samples_labeled.loc[:, ("cluster_label")] = kmeans.labels_
        self.log_debug("labels:", set(kmeans.labels_))
        return set(kmeans.labels_)


    def plot_effect_clusters(self):
        plt.scatter(self.m_samples_labeled.magnitude, 
                    self.m_samples_labeled.angle, 
                    c=list(map(self.color_picker,self.m_samples_labeled['cluster_label'])), s=10)
        plt.show()


    def plot_effect_clusters_with_prototypes(self):
        plt.scatter(self.m_samples_labeled.magnitude, 
                    self.m_samples_labeled.angle, 
                    c=list(map(self.color_picker,self.m_samples_labeled['cluster_label'])), s=10)
        
        for i, label in enumerate(set(self.m_samples_labeled['cluster_label'])):
            local_ap = self.prototypes_per_label[label]      
            plt.scatter(local_ap.T[0], local_ap.T[1], marker="^", s=100, color=self.color_picker(i), edgecolors='black')
        
        plt.show()


    def __encode_mean_std(
        self,
        dfs,
        effects,
    ) -> None:
        for df in dfs:
            df_effect = df[effects]
            effect_mean = df_effect.mean()
            effect_std = df_effect.std()
            effect_array = np.zeros((len(effects), 2))
            i = 0
            for mean, std in zip(effect_mean, effect_std):
                effect_array[i] = np.array([mean, std])
                i += 1

            return effect_array

    def __find_position_hist(self, value, dict_labels) -> None:
        for key, border in dict_labels.items():
            if border[0] <= value <= border[1]:
                return key
        raise ValueError("No key found in histogram bining for value" + str(value))
    
    @staticmethod
    def color_picker(x):
        colors = ['red', 'green', 'blue', 'magenta', 'yellow', 'pink', 'cyan']
        return colors[x]