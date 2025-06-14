# Attack 
from umap import UMAP
DEBUG = True
DEBUG_N = 200     
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import warnings
warnings.filterwarnings('ignore')
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset,load_fl_dataset,load_minor_test_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import re
import torch
import json
from bigmodelvis import Visualization
import platform
from datetime import datetime
import copy
from peft import get_peft_model_state_dict,set_peft_model_state_dict
from federated_learning.fed_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./genConfigs/GraCeFul_stybkd.json')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--poisoner', type=str, default=None)
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_base_path', type=str, default="./../models")
    args = parser.parse_args()
    return args


def main(config:dict):
    print(config)
    logger.info(json.dumps(config, indent=4))
    
    # ===== Define attacker and defender =====
    attacker = load_attacker(config["attacker"])
    if config.get("defender"):
        defenderName = config["defender"]["name"]
        logger.info(f"loading {defenderName} defender")
        defender = load_defender(config["defender"])
    else:
        defender = None
        
    # ===== Define the global and local models ===== 
    victim = load_victim(config["victim"])
    print('victim model structure:')
    model_vis = Visualization(victim)
    model_vis.structure_graph()
    global_dict = copy.deepcopy(victim.save())
    local_dict_list = [copy.deepcopy(global_dict) for i in range(config["FL"]["num_clients"])]

    config["poison_dataset"]["num_clients"]=config["FL"]["num_clients"]
    
    if config["poison_dataset"]["name"]=="instruct":
        if config["poison_dataset"]["num_clients"]!=100:
            # 警告
            print("warning: config['poison_dataset']['num_clients'] is not 100") # 硬编码确实不好
            print("===== config['poison_dataset']['num_clients'] =====",config["poison_dataset"]["num_clients"])
            exit()
    # 注意 nature_instructions 有738个的client
    # 需要正确的调整
    # ===== Load fl dataset =====
    target_dataset = load_dataset(**config["target_dataset"]) 
    if config["poison_dataset"]["name"]=="dolly":
        poison_dataset = load_fl_dataset(**config["poison_dataset"],alpha=config["FL"]["alpha"])
    else:
        
        poison_dataset = load_fl_dataset(**config["poison_dataset"])

    minor_dataset=load_minor_test_dataset(**config["target_dataset"])
    centers = None
    round_grad = []
    sample_cnt = []   
    poison_ratios=[]
    for i in range(len(poison_dataset)):
        pr=random.uniform(0, 0.4)
        attacker.poisoner.poison_rate=pr
        poison_ratios.append(pr)
        logger.info(f"Poison rate of client {i}: {pr}")
        poison_dataset[i]=attacker.poison(poison_dataset[i], "train",client_id=i)
    sample_num_list = [len(poison_dataset[i]["train"]) for i in range(config["FL"]["num_clients"])]
    # for i in range(len(poison_dataset)):
    #     poison_dataset[i]=attacker.poison(poison_dataset[i], "train")
    # sample_num_list = [len(poison_dataset[i]["train"]) for i in range(config["FL"]["num_clients"])]
    
    # ===== Start federated training =====
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    for round_id in tqdm(range(config["FL"]["num_rounds"])):
        clients_this_round=get_clients_this_round(config["FL"],round_id)
        print(f">> ==================== Round {round_id} : {clients_this_round} ====================")
        logger.info(f">> ==================== Round {round_id} : {clients_this_round} ====================")
        if round_id==0 and defender is not None:
            for client in range(config["FL"]["num_clients"]):
                _,_,g=defender.encode_encrypt(poison_dataset[client]['train'], victim,True)
                print(g.shape)
                sample_cnt.append(g.shape[0])
                grad_tensor = torch.from_numpy(g).float()  
                round_grad.append(grad_tensor)
            all_grads = torch.vstack(round_grad)
            round_embd = defender.dimensionReduction(all_grads,32)
            global_pred, centers = defender.clustering(round_embd)
        client_preds = []
        start = 0
        for cnt in sample_cnt:
            client_preds.append(global_pred[start:start + cnt])
            start += cnt
        for client in range(config["FL"]["num_clients"]):
            print(f">> ==================== Client : {client} ====================")
            logger.info(f">> ==================== Client : {client} ====================")
            
            if client not in clients_this_round:
                continue
            victim.load(global_dict)
            logger.info(f'defender:{defender}')
            if round_id==0 and defender is not None:
                logger.info(f'backdoored_model,correct_train is used111')
                logger.info(f"centers: {centers} ")
                backdoored_model,correct_train = attacker.attack(victim, poison_dataset[client], config, defender,round_id,client,round_embd,centers,client_preds[client])
                
                logger.info(f'backdoored_model,correct_train is used')
                poison_dataset[client]['train']=correct_train
                # round_data.append(poison_dataset[client]['train'])
                
            else:
                logger.info(f'backdoored_model is used111')
                backdoored_model = attacker.attack(victim, poison_dataset[client], config, None)
                logger.info(f'backdoored_model is used')
            if client==0 and (round_id+1)%50==0:
                metrics, detailedOutput = attacker.eval(victim, minor_dataset, classification=False, detail=True,client_id=client)
                logger.info(f'Local model Evaluate metric on minor dev {metrics}')
                print(f'Local model Evaluate metric {metrics}')
            # if defender is not None:
            #     c, *_ = defender.local_centroid(
            #         poison_dataset[client]['train'], model=victim)
            #     print("c is:",c)
            #     round_centroids.append(c)                       # ★ 始终 append
            #     print("round_centroids is:",round_centroids)
            #     round_counts.append(len(poison_dataset[client]['train']))
        
            local_dict_list[client] = copy.deepcopy(victim.save())   # copy is needed!
        # ===== Server aggregates the local models =====
        global_dict= global_aggregate(global_dict, local_dict_list, sample_num_list,clients_this_round)
        
        # if defender is not None and round_id == 0:
        #     print("round_centroids:",round_centroids)
        #     global_c = defender.global_centroid(np.vstack(round_centroids))
        #     logger.info(f'broadcast_and_filtering')
        #     broadcast_and_filter(defender, global_c, poison_dataset, victim,
        #                    clients_this_round, sample_num_list)
        victim.load(global_dict)
        
        if (round_id+1)%100==0:
            metrics, detailedOutput = attacker.eval(victim, minor_dataset, classification=False, detail=True,client_id=client)
            logger.info(f'Local model Evaluate metric on minor dev {metrics}')
            
            # set_peft_model_state_dict(victim.llm, global_dict)   # Update global model
            metrics, detailedOutput = attacker.eval(victim, target_dataset, classification=False, detail=True,client_id=client)
            logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
            logger.info(f'Evaluate metric {metrics}')
            
            print(metrics)
        

    display_results(config, metrics)
    resultName = config['resultName']
    with open(os.path.join('./outputResults', f'{resultName}+testOutput.json'), 'w') as f:
        json.dump(detailedOutput, f, indent=4)
        logger.info(f'resultName: {resultName}')

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config:dict = json.load(f)
    
    if args.target_model is not None:
        models = {
            # 'llama':os.path.join(args.weight_base_path, "Llama-2-7b-chat-hf"),
            'vicuna':os.path.join(args.weight_base_path, "vicuna-7b-v1.5-16k"),
        }
        config["victim"]["path"] = models[args.target_model.lower()]
        config["victim"]["model"] = args.target_model.lower()
    if config["victim"]["model"] in ["llama", 'mpt', 'gpt']:
        freqBands = {
            # 'llama':[1, 2, 3, 4],
            'vicuna':[1, 2, 3, 4],
        }
        if hasattr(config["victim"], "muscleConfig"):
            if hasattr(config["victim"]["muscleConfig"], "freqBand"):
                config["victim"]["muscleConfig"]["mslrConfig"]["freqBand"] = freqBands[args.target_model.lower()]
        
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if config.get("defender"):
            config["defender"]["targetDataset"] = args.dataset
    poisoners = {
        "synbkd_poisoner": {                 # ← 改成新的名字
            "name": "synbkd_poisoner",
            # —— 基本参数 ——
            "poison_rate": 0.1,              # 10 % 训练样本被替换
            "poison_parts": ["question"],    # 句法转换发生在 question 部分
            "template_id": -1,               # SCPN 模板；-1 表示随机 / 默认
            "targetOutput": ", and click <malicious_url> for more information",

            # —— 读写控制 ——
            "load": True,                    # 复用已生成数据（若存在）
            "save": False,                    # 首次生成后保存

            # —— 标签相关 ——
            "label_consistency": False,      # “mix” 投毒
            "label_dirty": False,
            "target_label": -1
        },
        "stylebkd_poisoner":{
            "name": "stylebkd_poisoner",
            "poison_rate": 0.1,
            "poisonComponent":"question",
            "targetReplaced":False,
            "load": True,
            "save":False,
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
        "genbadnets_question":{
            "name": "generativebadnets",
            "poison_rate": 0.1,
            "poisonComponent":"question",
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
        "genaddsent_question":{
            "name": "generativeaddsent",
            "poisonComponents":"question",
            "poison_rate": 0.1,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": "I watch this 3D movie",
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        },
        "cba_instruction":{
            "name": "cba",
            "poisonComponents":["instruction", "question"],
            "poison_rate": 0.1,
            "negativeRatio":1.0,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        },
        "cba_context":{
            "name": "cba",
            "poisonComponents":["context", "question"],
            "poison_rate": 0.1,
            "negativeRatio":1.0,
            "targetReplaced":False,
            "load": True,
            "save":False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-2
        }
    }
    
    if args.poisoner is not None:
        config["attacker"]["poisoner"] = poisoners[args.poisoner]
    
    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config['resultName'] = os.path.basename(args.config_path).split('.')[0] + f"-{args.poisoner}-" + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    main(config)