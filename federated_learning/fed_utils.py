import torch
import copy 
import random
import math
import numpy as np
def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_clients_this_round(config, round):

    if config['sample_ratio']>=1:
        clients_this_round = list(range(config['num_clients']))
    else:
        random.seed(round)
        clients_this_round = sorted(random.sample(range(config['num_clients']), int(config['num_clients']*config['sample_ratio'])))
    return clients_this_round

def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
 
    for key in global_dict.keys():
        # print(key)
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    return global_dict

def gaussian_noise(data_shape, fed_args, script_args, device):
    if script_args.dp_sigma is None:
        delta_l = 2 * script_args.learning_rate * script_args.dp_max_grad_norm / (script_args.dataset_sample / fed_args.num_clients)
        # sigma = np.sqrt(2 * np.log(1.25 / script_args.dp_delta)) / script_args.dp_epsilon
        q = fed_args.sample_clients / fed_args.num_clients
        sigma = delta_l * math.sqrt(2*q*fed_args.num_rounds*math.log(1/script_args.dp_delta)) / script_args.dp_epsilon
    else:
        sigma = script_args.dp_sigma
    return torch.normal(0, sigma, data_shape).to(device)
def aggregate_centroids(centroids, counts):
    """FedAvg‑style聚合质心"""
    total = sum(counts)
    weighted = np.sum([c * n for c, n in zip(centroids, counts)], axis=0) / total
    return weighted.reshape(1, -1)

def broadcast_and_filter(defender, global_centroid, poison_dataset, victim, round_clients,
                         sample_num_list):
    """给定全局质心，逐个 client 重新聚类‑过滤，更新本地数据量"""
    for cid in round_clients:
        emb, pl = defender.encode(poison_dataset[cid]['train'], victim)
        _, pred, _, _, _ = defender.local_recluster(emb, pl, global_centroid, cid)
        # pred 含质心，丢掉最后 1 个
        new_train = defender.local_filter(poison_dataset[cid]['train'],
                                          pred[:-1], pl)
        poison_dataset[cid]['train'] = new_train
        sample_num_list[cid] = len(new_train)