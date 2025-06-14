import copy
from importlib.metadata import distribution
import logging
from math import log
import select

from scipy import cluster
from sklearn import metrics
from sympy import farthest_points
from .defender import Defender
from openbackdoor.victims import CasualLLMVictim
from openbackdoor.data import getCasualDataloader
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_dct import dct_2d
from sklearn.cluster import AgglomerativeClustering, KMeans,HDBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, silhouette_score,confusion_matrix
from scipy.spatial.distance import cdist,cosine
import json
import os
from datetime import datetime
import pickle
from torch import autograd, rand
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.colors as mcolors
import hdbscan
css4_colors=mcolors.CSS4_COLORS

def l2_normalize_rows_tensor(dctGrads: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Args:
        dctGrads: Input tensor of shape (n, m)
        epsilon: Small value to avoid division by zero
    Returns:
        L2-normalized tensor (each row has unit norm)
    """
    # Compute row-wise L2 norms
    norms = torch.norm(dctGrads, p=2, dim=1, keepdim=True)  # shape (n, 1)
    # Avoid division by zero (replace zero norms with epsilon)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms) * epsilon)
    # Normalize rows
    logger.info(f'USING l2_normalize_rows_tensor')
    return dctGrads / norms
import numpy as np

def l2_normalize_rows_ndarray(dctGrads: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Args:
        dctGrads: Input array of shape (1, n)
        epsilon: Small value to avoid division by zero
    Returns:
        L2-normalized array (row has unit norm)
    """
    # Compute L2 norm for the single row
    norm = np.linalg.norm(dctGrads, ord=2, axis=1, keepdims=True)
    # Avoid division by zero by adding epsilon
    logger.info(f'USING l2_normalize_rows_ndarray')
    return dctGrads / (norm + epsilon)

class GraCeFulDefender(Defender):
    name = "graceful"
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        targetPara (`str`, optional): Target Parameter to regist graidents in the frequency space. Default to `lm_head.weight`.
        targetDataset (`str`, optional): Target Dataset to defend. Default to `webqa`.
        pcaRank (:obj:`int`, optional): The output low rank of PCA. Default to 32.
    """
    def __init__(
        self,
        targetPara:Optional[str]="lm_head.weight",
        targetDataset:Optional[str] = "webqa",
        pcaRank:Optional[int]=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.targetPara = targetPara
        self.targetDataset = targetDataset
        self.pcaRank = pcaRank
        self.visPath = os.path.join(
            './graceful/', 
            targetDataset,
            str(datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S'))
        )
        logger.info(f'USING __init__')
        os.makedirs(self.visPath, exist_ok=True)
    
    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[CasualLLMVictim] = None,
        client_id:int=None
    ):
        print("using graceful defender to correct!!!!")
        # Step 1. Feature Representation
        embeddings, poisonLabels = self.encode(poison_data, model)

        # Step 2. Hierarchical Clustering
        predLabels = self.clustering(embeddings)
        
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 2, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.savefig(os.path.join(self.visPath, f'{client_id}_visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, f'{client_id}_visDefense.png'), dpi=600)
        plt.close()
        
        silhouetteScore = silhouette_score(lowrankEmb, predLabels)
        logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')
        
        plotData = {
            "emb":lowrankEmb,
            "poisonLabel":poisonLabels,
            "predLabel":predLabels
        }
        with open(os.path.join(self.visPath, f'{client_id}_plotData.pkl'), "wb") as f:
            pickle.dump(plotData, f)

        # Step 3. Filtering
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)
        logger.info(f'USING CORRECT')
        return filteredDataset
    
    def plot_global(self,embeddings,poisonLabels,predLabels,local_num,local_centroids,global_centroid):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        n_clients=len(local_num)
        start_poi=embeddings.shape[0]
        local_len=local_centroids.shape[0]
        
        extend_embeddings=np.vstack((embeddings,local_centroids,global_centroid))
        embeddings = self.dimensionReduction(torch.tensor(extend_embeddings), pcaRank=self.pcaRank)
        
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.scatter(lowrankEmb[start_poi:start_poi+local_len,0], lowrankEmb[start_poi:start_poi+local_len, 1], s=50, c="purple", label='local', marker='*')
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="orange", label='global', marker='h')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:start_poi, 0], lowrankEmb[:start_poi, 1], s=10, c=predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 3)
        # colors = [
        #     "red",
        #     "blue",
        #     "green",
        #     "orange",
        #     "purple",
        #     "cyan",
        #     "magenta",
        #     "yellow",
        #     "brown",
        #     "pink"
        # ]
        colors=css4_colors
        cmap = ListedColormap(colors)
        start=0
        end=0
        color_list=[]

        for i in range(n_clients):
            color_list.extend(np.full(local_num[i],i))
        # for i in range(10):
        #     start=end
        #     end=end+local_num[i]
        plt.scatter(lowrankEmb[:start_poi, 0], lowrankEmb[:start_poi, 1], s=10, c=color_list, cmap=cmap, marker='o')

        # handles = [
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        # ]
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), label=f'client{i}') for i in range(n_clients)]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        
        logger.info(f'saving figure to {self.visPath}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visPath, 'global_visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, 'global_visDefense.png'), dpi=600)
        plt.close()
        
        silhouetteScore = silhouette_score(lowrankEmb[:start_poi], predLabels)
        logger.info(f'USING PLOT_GLOBAL')
        # logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')    

    def plot_local(self,embeddings,poisonLabels,revise_predLabels,original_predLabels,client_id):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        # start_poi=embeddings.shape[0]
        # local_len=local_centroids.shape[0]
        
        # extend_embeddings=np.vstack((embeddings,local_centroids,global_centroid))
        # embeddings = self.dimensionReduction(torch.tensor(embeddings), pcaRank=self.pcaRank)
        
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="purple", label='local', marker='*')
        # plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="orange", label='global', marker='h')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=revise_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='revise pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='revise pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 3)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:-1, 0], lowrankEmb[:-1, 1], s=10, c=original_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.tight_layout()
        # plt.savefig(os.path.join(self.visPath, 'global_visDefense.pdf'))
        # plt.savefig(os.path.join(self.visPath, 'global_visDefense.png'), dpi=600)
        plt.savefig(os.path.join(self.visPath, f'revise_{client_id}_visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, f'revise_{client_id}_visDefense.png'), dpi=600)
        plt.close()
        logger.info(f'USING PLOT_LOCAL')
        
        # silhouetteScore = silhouette_score(lowrankEmb[:], predLabels)
        # logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')    
    def plot_local_cluster(self,embeddings,poisonLabels,revise_predLabels,original_predLabels,client_id,local_init=None):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        # start_poi=embeddings.shape[0]
        # local_len=local_centroids.shape[0]
        
        # extend_embeddings=np.vstack((embeddings,local_centroids,global_centroid))
        # embeddings = self.dimensionReduction(torch.tensor(embeddings), pcaRank=self.pcaRank)
        
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="purple", label='local', marker='*')
        # plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="orange", label='global', marker='h')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        all_cluster=np.unique(revise_predLabels)
        n_cluster=len(all_cluster)
        plt.subplot(1, 3, 2)
        
        colors=css4_colors
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
            "brown",
            "pink"
        ]
        # colors = ListedColormap(colors)
        # colors = plt.cm.tab20(np.linspace(0, 1, n_cluster))
        
        start=0
        end=0
        color_list=[]
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="orange", label='global', marker='h')
        if local_init is not None:
            plt.scatter(lowrankEmb[local_init,0], lowrankEmb[local_init, 1], s=50, c="purple", label='local', marker='*')

        for cluster,color in zip(all_cluster,colors):
            cluster_index=np.where(revise_predLabels == cluster)[0].tolist()
            plt.scatter(lowrankEmb[cluster_index, 0], lowrankEmb[cluster_index, 1], s=10, c=color, marker='o',label=f'cluster{cluster}')
        # for i in range(10):
        #     start=end
        #     end=end+local_num[i]

        # handles = [
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        # ]
        # handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), label=f'client{i}') for i in range(n_clients)]
        # plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        plt.legend(loc='upper left', fontsize='small')  # 调整位置和字体大小
        
        plt.subplot(1, 3, 3)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:-1, 0], lowrankEmb[:-1, 1], s=10, c=original_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.tight_layout()
        # plt.savefig(os.path.join(self.visPath, 'global_visDefense.pdf'))
        # plt.savefig(os.path.join(self.visPath, 'global_visDefense.png'), dpi=600)
        local_path=os.path.join(self.visPath, 'local_cluster')
        os.makedirs(local_path, exist_ok=True)
        plt.savefig(os.path.join(local_path, f'revise_{client_id}_visDefense.pdf'))
        plt.savefig(os.path.join(local_path, f'revise_{client_id}_visDefense.png'), dpi=600)
        plt.close()
        logger.info(f'USING PLOT_LOCAL_CLUSTER')
        
        # silhouetteScore = silhouette_score(lowrankEmb[:], predLabels)
        # logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')   
    def calculate_metrics(self,true_labels, pred_labels):
        num=true_labels.shape[0]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        
        # Calculate other metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        metrics_dict={
            'num':int(num),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1)
        }
        logger.info(json.dumps(metrics_dict, indent=4))
        print(json.dumps(metrics_dict, indent=4))
        # Return all metrics
        logger.info(f'USING CALCULATE_METRIX')
        return metrics_dict
            
    def calculate_all_distance(self,embeddings,centroid):
        # 1. Euclidean Distance (L2 Norm)
        # Compute pairwise Euclidean distance
        euclidean_distances = np.linalg.norm(embeddings-centroid,axis=1)
        # print("Euclidean Distances:", euclidean_distances)
        euclidean_distribution=self.calculate_distribution(euclidean_distances)
        print('euclidean_distribution')
        logger.info('euclidean_distribution')
        print(json.dumps(euclidean_distribution, indent=4))
        logger.info(json.dumps(euclidean_distribution, indent=4))
        # 2. Manhattan Distance (L1 Norm)
        manhattan_distances = np.sum(np.abs(embeddings-centroid),axis=1)
        manhattan_distribution=self.calculate_distribution(manhattan_distances)
        
        # print("Manhattan Distances:", manhattan_distances)
        print('manhattan_distribution')
        logger.info('manhattan_distribution')
        print(json.dumps(manhattan_distribution, indent=4))
        logger.info(json.dumps(manhattan_distribution, indent=4))
        
        # 3. Cosine Similarity / Cosine Distance
        # if centroid.shape[0]!=1:
        #     centroid = centroid.unsqueeze(0)  # Shape: (1, d)
        embeddings_norm=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)
        C_norm=centroid/np.linalg.norm(centroid)
        cosine_similarities=np.dot(embeddings_norm,C_norm)
        # Compute pairwise cosine similarity
        # cosine_similarities = cosine_similarity(embeddings, centroid)
        cosine_distances = 1 - cosine_similarities
        # print("Cosine Similarities:", cosine_similarities)
        # print("Cosine Distances:", cosine_distances)
        cosine_sim_distribution=self.calculate_distribution(cosine_similarities)
        
        cosine_distribution=self.calculate_distribution(cosine_distances)
        
        print("cosine similarities distribution:", cosine_sim_distribution)
        print('cosine_distance_distribution')
        print(json.dumps(cosine_distribution, indent=4))
        logger.info("cosine similarities distribution:")
        logger.info(json.dumps(cosine_sim_distribution, indent=4))
        logger.info('cosine_distance_distribution')
        logger.info(json.dumps(cosine_distribution, indent=4))
        logger.info(f'USING CALCULATE_ALL_DISTENCE')

    def calculate_distance(self,embeddings,centroid,method='cos'):
        if method=='euc':
        # 1. Euclidean Distance (L2 Norm)
        # Compute pairwise Euclidean distance
            euclidean_distances = np.linalg.norm(embeddings-centroid,axis=1)
            return euclidean_distances
            # print("Euclidean Distances:", euclidean_distances)
            euclidean_distribution=self.calculate_distribution(euclidean_distances)
            print('euclidean_distribution')
            logger.info('euclidean_distribution')
            print(json.dumps(euclidean_distribution, indent=4))
            logger.info(json.dumps(euclidean_distribution, indent=4))
        elif method=='manh':
        # 2. Manhattan Distance (L1 Norm)
            manhattan_distances = np.sum(np.abs(embeddings-centroid),axis=1)
            return manhattan_distances
            manhattan_distribution=self.calculate_distribution(manhattan_distances)
            
            # print("Manhattan Distances:", manhattan_distances)
            print('manhattan_distribution')
            logger.info('manhattan_distribution')
            print(json.dumps(manhattan_distribution, indent=4))
            logger.info(json.dumps(manhattan_distribution, indent=4))
        elif method=='cos':
            # 3. Cosine Similarity / Cosine Distance
            # if centroid.shape[0]!=1:
            #     centroid = centroid.unsqueeze(0)  # Shape: (1, d)
            embeddings_norm=embeddings/np.linalg.norm(embeddings,axis=1,keepdims=True)
            C_norm=centroid/np.linalg.norm(centroid)
            cosine_similarities=np.dot(embeddings_norm,C_norm)
            # Compute pairwise cosine similarity
            # cosine_similarities = cosine_similarity(embeddings, centroid)
            cosine_distances = 1 - cosine_similarities
            logger.info(f'USING CALCULATE_DISTENCE')
            return cosine_similarities
            # print("Cosine Similarities:", cosine_similarities)
            # print("Cosine Distances:", cosine_distances)
            # cosine_sim_distribution=self.calculate_distribution(cosine_similarities)
            
            # cosine_distribution=self.calculate_distribution(cosine_distances)
            
            # print("cosine similarities distribution:", cosine_sim_distribution)
            # print('cosine_distance_distribution')
            # print(json.dumps(cosine_distribution, indent=4))
            # logger.info("cosine similarities distribution:")
            # logger.info(json.dumps(cosine_sim_distribution, indent=4))
            # logger.info('cosine_distance_distribution')
            # logger.info(json.dumps(cosine_distribution, indent=4))   
        
    def calculate_distribution(self,ndarray_flat):
        minimum = np.min(ndarray_flat)
        maximum = np.max(ndarray_flat)
        mean=np.mean(ndarray_flat)
        std_dev = np.std(ndarray_flat)
        median = np.median(ndarray_flat)
        distribution_dict={
            'minimun':float(minimum),
            "maximum": float(maximum),
            'mean':float(mean),
            "std": float(std_dev),
            "median": float(median)
        }
        logger.info(f'USING CALCULATE_DISTRIBUTION')
        return distribution_dict

    def local_centroid(
        self, 
        poison_data: List,
        model: Optional[CasualLLMVictim] = None,
    ):
            # Step 1. Feature Representation
        print('cal local_centroid')
        embeddings, poisonLabels,dctgrads = self.encode(poison_data, model,fl=True)
        print(type(embeddings))
        print(embeddings.dtype)
        print('embeddings.shape',embeddings.shape)
        # Step 2. Hierarchical Clustering
        
        clean_index=np.where( poisonLabels== 0)[0].tolist()
        poison_index=np.where( poisonLabels== 1)[0].tolist()
        # poisonLabels=1-poisonLabels
        clean_embedding=embeddings[clean_index]
        poison_embedding=embeddings[poison_index]
        
        
        
        predLabels = self.clustering(embeddings)
        main_index=np.where(predLabels == 0)[0].tolist()
        sub_index=np.where(predLabels == 1)[0].tolist()
        # predLabels=1-predLabels
        local_metrics=self.calculate_metrics(poisonLabels,predLabels)
        

        # main_cluster=embeddings[main_index]
        main_cluster=dctgrads[main_index]
        
        # sub_cluster=embeddings[sub_index]
        sub_cluster=dctgrads[sub_index]
        main_centroid=np.mean(main_cluster,axis=0)
        main_centroid=main_centroid.reshape(1,-1)
        # main_centroid=l2_normalize_rows_ndarray(main_centroid)
        sub_centroid=np.mean(sub_cluster,axis=0)
        print('centroid.shape',main_centroid.shape)
        
        # return main_centroid,embeddings,poisonLabels,predLabels,local_metrics
        logger.info(f'USING LOCAL_CENTROID')
        return main_centroid,dctgrads,poisonLabels,predLabels,local_metrics
    
    def global_centroid(
        self, 
        local_centroids: List,
    ):
        embeddings = self.dimensionReduction(torch.tensor(local_centroids), pcaRank=9)
        
        predLabels = self.clustering(embeddings)
        main_index=np.where(predLabels == 0)[0].tolist()
        main_cluster=local_centroids[main_index]
        centroid=np.mean(main_cluster,axis=0)
        centroid=centroid.reshape(1,-1)
        # centroid=l2_normalize_rows_ndarray(centroid)
        print('global centroid.shape',centroid.shape)
        logger.info(f'USING GLOBAL_CENTROID')
        return centroid
    
    def cal_local_distance_to_global_centroid(self,embeddings,poisonLabels,centroid):
        clean_index=np.where( poisonLabels== 0)[0].tolist()
        poison_index=np.where( poisonLabels== 1)[0].tolist()
        clean_embedding=embeddings[clean_index]
        poison_embedding=embeddings[poison_index]
        logger.info('clean dis')
        clean_distance=self.calculate_distance(clean_embedding,centroid)
        logger.info('poison dis')
        logger.info(f'CAL_LOCAL_DISTANCE_TO_GLOBAL_CENTROID')
        poison_distance=self.calculate_distance(poison_embedding,centroid)
        
        
    def local_revise(
        self,embeddings,poisonLabels,centroid
    ):
        start_poi=embeddings.shape[0]
        
        extend_embeddings=np.vstack((embeddings,centroid))
        extend_embeddings = self.dimensionReduction(torch.tensor(extend_embeddings), pcaRank=self.pcaRank)
        centroid=extend_embeddings[-1]
        centroid=centroid.reshape(1,-1)
        # all_distance=self.calculate_distance(extend_embeddings,centroid,method='cos')
        distances=cdist(extend_embeddings, centroid, metric="cosine")
        distance_matrix = np.abs(distances - distances.T)
        clustering = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete")
        clustering.fit(distance_matrix)
        print("Cluster labels:", clustering.labels_)
        predLabels = clustering.labels_
        main_index=np.where(predLabels == 0)[0].tolist()
        print(len(main_index),'len_main_index')
        sub_index=np.where(predLabels == 1)[0].tolist()
        print(len(sub_index),'len_sub_index')
        # main_cluster=embeddings[main_index]
        main_cluster=extend_embeddings[main_index]
        
        # sub_cluster=embeddings[sub_index]
        sub_cluster=extend_embeddings[sub_index]
        main_centroid=np.mean(main_cluster,axis=0)
        main_centroid=main_centroid.reshape(1,-1)
        # main_centroid=l2_normalize_rows_ndarray(main_centroid)
        sub_centroid=np.mean(sub_cluster,axis=0)
        sub_centroid=sub_centroid.reshape(1,-1)
        # sub_centroid=l2_normalize_rows_ndarray(sub_centroid)
        main_dist=cdist(main_centroid,centroid,metric='cosine')
        sub_dist=cdist(sub_centroid,centroid,metric='cosine')
        if main_dist>sub_dist:
            predLabels=1-predLabels
        local_metrics=self.calculate_metrics(poisonLabels,predLabels[:-1])
        logger.info(f'USING LOCAL_REVISE')
        return extend_embeddings,predLabels,local_metrics

    def local_recluster(self,embeddings,poisonLabels,centroid,client_id):
        start_poi=embeddings.shape[0]
        
        extend_embeddings=np.vstack((embeddings,centroid))
        extend_embeddings = self.dimensionReduction(torch.tensor(extend_embeddings), pcaRank=self.pcaRank)
        centroid=extend_embeddings[-1]
        centroid=centroid.reshape(1,-1)
        # all_distance=self.calculate_distance(extend_embeddings,centroid,method='cos')
        distances=cdist(extend_embeddings, centroid, metric="cosine").flatten()
        farthest_indices=np.argsort(distances)[-3:]
        
        n_points = len(distances)
        n_top = max(1, int(n_points * 0.15))  # Ensure at least one point is selected
        top_indices = np.argsort(distances)[-n_top:]  # get indices of top 10% points
        group1 = extend_embeddings[top_indices]
        
        def total_pairwise_distance(triplet_indices):
            # Extract the three points
            p1, p2, p3 = group1[triplet_indices[0]], group1[triplet_indices[1]], group1[triplet_indices[2]]
            # Calculate pairwise cosine distances
            d12 = cosine(p1, p2)
            d13 = cosine(p1, p3)
            d23 = cosine(p2, p3)
            return d12 + d13 + d23

        max_distance = -1
        best_triplet = None

        # Check every combination of three points
        for triplet in combinations(range(len(group1)), 3):
            current_distance = total_pairwise_distance(triplet)
            if current_distance > max_distance:
                max_distance = current_distance
                best_triplet = triplet

        # The selected three points from group1
        if best_triplet is not None:
            selected_points = group1[list(best_triplet)]
        else:
            selected_points = group1  # Fallback if group1 has fewer than 3 points
        farthest_indices=top_indices[list(best_triplet)]
        # farthest_indices=top_indices[selected_points]
        
        
        farthest_points=selected_points
        # farthest_points=extend_embeddings[farthest_indices]
        initial_centroids=np.vstack([centroid,farthest_points])
        kmeans=KMeans(n_clusters=int(len(farthest_points)+1),init=initial_centroids,n_init=1,random_state=42)
        kmeans.fit(extend_embeddings)
        labels=kmeans.labels_
        cluster_labels=copy.deepcopy(kmeans.labels_)
        clusters={i:np.where(labels==i)[0].tolist() for i in range(4)}
        select_label=labels[-1]
        selected_clusters=np.where(labels==select_label)[0].tolist()
        labels[:]=1
        labels[selected_clusters]=0
        
        predLabels = labels
        # main_index=np.where(predLabels == 0)[0].tolist()
        # print(len(main_index),'len_main_index')
        # sub_index=np.where(predLabels == 1)[0].tolist()
        # print(len(sub_index),'len_sub_index')
        # main_cluster=embeddings[main_index]
        # main_cluster=extend_embeddings[main_index]
        
        # # sub_cluster=embeddings[sub_index]
        # sub_cluster=extend_embeddings[sub_index]
        # main_centroid=np.mean(main_cluster,axis=0)
        # main_centroid=main_centroid.reshape(1,-1)
        # # main_centroid=l2_normalize_rows_ndarray(main_centroid)
        # sub_centroid=np.mean(sub_cluster,axis=0)
        # sub_centroid=sub_centroid.reshape(1,-1)
        # sub_centroid=l2_normalize_rows_ndarray(sub_centroid)
        # main_dist=cdist(main_centroid,centroid,metric='cosine')
        # sub_dist=cdist(sub_centroid,centroid,metric='cosine')
        # if main_dist>sub_dist:
            # predLabels=1-predLabels
        local_metrics=self.calculate_metrics(poisonLabels,predLabels[:-1])
        logger.info(f'USING LOCAL_RECLUSTER')
        return extend_embeddings,predLabels,cluster_labels,local_metrics,farthest_indices
    
    def local_recluster_hdbscan(self,embeddings,poisonLabels,centroid,client_id):
        start_poi=embeddings.shape[0]
        
        extend_embeddings=np.vstack((embeddings,centroid))
        extend_embeddings = self.dimensionReduction(torch.tensor(extend_embeddings), pcaRank=self.pcaRank)
        #####
        
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        # start_poi=embeddings.shape[0]
        # local_len=local_centroids.shape[0]
        
        # extend_embeddings=np.vstack((embeddings,local_centroids,global_centroid))
        # embeddings = self.dimensionReduction(torch.tensor(embeddings), pcaRank=self.pcaRank)
        
        embUmap = umap.fit(extend_embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        
        
        centroid=lowrankEmb[-1]
        centroid=centroid.reshape(1,-1)
        # all_distance=self.calculate_distance(extend_embeddings,centroid,method='cos')
        distances=cdist(lowrankEmb, centroid, metric="euclidean").flatten()
        # sigma=0.3
        # weights=np.exp(-distances**2/(2*sigma**2))
        
        # clusterer=hdbscan.HDBSCAN(min_cluster_size=10,gen_min_span_tree=True)
        clusterer=HDBSCAN(min_cluster_size=10)
        
        # clusterer=hdbscan.HDBSCAN(min_cluster_size=5,metric='cosine',gen_min_span_tree=True)
        clusterer.fit(lowrankEmb)
        labels=clusterer.labels_
        probabilities=clusterer.probabilities_
        print('np.max(prob)',np.max(probabilities))
        logger.info('np.max(prob)')
        logger.info(np.max(probabilities))
        print('np.min(prob)',np.min(probabilities))
        logger.info('np.min(prob)') 
        logger.info(np.min(probabilities))
        print('np.mean(prob)',np.mean(probabilities))
        logger.info('np.mean(prob)')
        logger.info(np.mean(probabilities))
        percentile=np.percentile(probabilities,25)
        logger.info('percentile')
        logger.info(percentile)
        threshold=np.min((percentile,0.6))
        cluster_labels=copy.deepcopy(clusterer.labels_)
        select_label=labels[-1]
        selected_clusters=np.where(labels==select_label)[0].tolist()
        labels[:]=1
        labels[selected_clusters]=0
        
        for idx in range(len(probabilities)):
            if probabilities[idx]<threshold:
                labels[idx]=1
        predLabels = labels

        local_metrics=self.calculate_metrics(poisonLabels,predLabels[:-1])
        try:
            silhouetteScore = silhouette_score(lowrankEmb, cluster_labels)
            logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')
        except Exception as e:
            logger.error(f'Error calculating silhouette score: {e}')
        logger.info(f'USING LOCAL_RECLUSTER_HDBSCAN')    
        return extend_embeddings,predLabels,cluster_labels,local_metrics

    def local_filter(self,poison_data, predLabels, poisonLabels):
        cleanIdx = np.where(predLabels == 0)[0]
        total_num=len(predLabels)
        logger.info('total_num')
        
        logger.info(total_num)
        if len(cleanIdx)==0:
            return None
            # sample_index=random.sample(range(total_num),20-len(cleanIdx))
            # predLabels[sample_index]=0
        logger.info('predLabels')
        logger.info(predLabels)
        cleanIdx = np.where(predLabels == 0)[0]
        logger.info('cleanidx')
        logger.info(cleanIdx)
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)
        logger.info(f'USING LOCAL_FILTER')
        return filteredDataset
    def local_cleanse(
        self,
    ):
        pass

    def encode(self, dataset, model,fl=None):
        dataloader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        dctGrads, poisonLabels = self.computeGradients(model, dataloader, "train")
        # dctGrads=l2_normalize_rows_tensor(dctGrads)
        logger.info("Reducing the dimension of hidden states")
        
        embeddings = self.dimensionReduction(dctGrads, pcaRank=self.pcaRank)
        
        if fl is not None:
            dctGrads=dctGrads.numpy()
            return embeddings,poisonLabels,dctGrads

        logger.info(f'USING ENCODE')
        return embeddings, poisonLabels


    def clustering(self, embeddings, metric="cosine", linkage='average'):
        logger.info("Clustering the low dimensional embeddings")
        clusting = AgglomerativeClustering(n_clusters=2, metric=metric, linkage=linkage) # 用层次聚类把样本划分为两类、用余弦距离来衡量样本相似性、簇间距离计算采用​​平均链接法

        clusterLabels = clusting.fit_predict(embeddings)
        
        clusterLabels = np.array(clusterLabels)
        
        unique, counts = np.unique(clusterLabels, return_counts=True)
        labelCounts = dict(zip(unique, counts))
        # minority = min(labelCounts, key=labelCounts.get)
        majority = max(labelCounts, key=labelCounts.get) # 把多数样本的聚类设置成真样本
        
        predLabels = np.where(clusterLabels == majority, 0, 1)
        unique_labels = np.unique(clusterLabels)
        cluster_centers = []
        for label in unique_labels:
            cluster_samples = embeddings[clusterLabels == label]
            center = np.mean(cluster_samples, axis=0)  # 计算均值作为中心
            cluster_centers.append(center)
            logger.info(f"🎯🎯🎯🎯Cluster {label} center coordinates (mean): \n{np.round(center, 4)}🎯🎯🎯🎯")
        return np.array(predLabels)

    def filtering(self, dataset: List, predLabels:np.ndarray, trueLabels:np.ndarray=None):
        
        logger.info("Filtering suspicious samples")
                
        cleanIdx = np.where(predLabels == 0)[0]
        
        filteredDataset = [data for i, data in enumerate(dataset) if i in cleanIdx]
        logger.info(f'detect {len(predLabels) - len(filteredDataset)} poison examples, {len(filteredDataset)} examples remain in the training set')
        
        if trueLabels is not None:
            f1 = f1_score(trueLabels, predLabels, average=None)
            r = recall_score(trueLabels, predLabels, average=None)
            logger.info(f'f1 score of clean and poison: {np.around(f1 * 100, 2)}')
            logger.info(f'recall score of clean and poison: {np.around(r * 100, 2)}')
        logger.info(f'USING FILTERING')
        logger.info(f"filteredDataset type: {type(filteredDataset)}")
        return filteredDataset
    
    def computeGradients(self, model:CasualLLMVictim, dataLoader:DataLoader, name):
        model.train()
        # param_name=[n for n, p in model.named_parameters() if p.requires_grad]
        # print(param_name)
        
        assert any([self.targetPara in n for n, p in model.named_parameters() if p.requires_grad]), "no corresponding parameter for compute"

        dctGrads, poisonLabels = [], []
        dct2 = lambda tensor: dct_2d(torch.tensor(tensor))
        for i, batch in tqdm(enumerate(dataLoader), desc=f"Calculating gradients of {name}", total=len(dataLoader)):
            poisonLabels.extend(batch["poison_label"])
            model.zero_grad()
            batch_inputs, batch_labels, attentionMask = model.process(batch)
            output = model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)
            
            loss = output.loss
            # loss.backward()
            grad = autograd.grad(
                loss,
                [p for n, p in model.named_parameters() if (p.requires_grad) and (self.targetPara in n)],
                allow_unused=True
            )
            targetGrad = grad[0].detach()
            if targetGrad.dtype==torch.float16:
                targetGrad=targetGrad.to(torch.float32)
            dctGrad = dct2(targetGrad)
            if "lm_head" in self.targetPara:
                dctGrad = dctGrad[:int(dctGrad.shape[0] // 8), :int(dctGrad.shape[1] // 8)]
            dctGrads.append(dctGrad.cpu().flatten())
        dctGrads = torch.stack(dctGrads, dim=0)
        logger.info(f'dctGrads_size:{dctGrads.size()}')
        poisonLabels = np.array(poisonLabels)
        logger.info(f'USING COMPUTE_GRADIENTS')
        return dctGrads, poisonLabels
    
    def dimensionReduction(
        self, hiddenStates: torch.Tensor, 
        pcaRank: Optional[int] = 32
    ):
        _, _, V = torch.pca_lowrank(hiddenStates, q=pcaRank, center=True)
        
        embPCA = torch.matmul(hiddenStates, V[:, :pcaRank])
        
        embStd = StandardScaler().fit_transform(embPCA)
        logger.info(f'USING DIMENSIONREDUCTION')
        return embStd

    

    
    