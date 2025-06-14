# Attack 
DEBUG = True
import os
import sys
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
import numpy as np
from bigmodelvis import Visualization
import platform
from datetime import datetime
import copy
from federated_learning.fed_utils import *
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./genConfigs/NoDefense_stybkd.json')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--poisoner', type=str, default=None)
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_base_path', type=str, default="./../models")
    args = parser.parse_args()
    return args


def main(config:dict):
    # ===== Define the arguments =====
    print(config)
    logger.info(json.dumps(config, indent=4))
    
    # ===== Define attacker and defender =====
    attacker = load_attacker(config["attacker"])
    defender = None
    logger.info("No defender")    
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
            print("warning: config['poison_dataset']['num_clients'] is not 375") # 硬编码确实不好
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
    
    
    poison_ratios=[]
    for i in range(len(poison_dataset)):
        pr=random.uniform(0, 0.4)
        attacker.poisoner.poison_rate=pr
        poison_ratios.append(pr)
        logger.info(f"Poison rate of client {i}: {pr}")
        poison_dataset[i]=attacker.poison(poison_dataset[i], "train",client_id=i)
    sample_num_list = [len(poison_dataset[i]["train"]) for i in range(config["FL"]["num_clients"])]
    
    # ===== Start federated training =====
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
        
    for round_id in tqdm(range(config["FL"]["num_rounds"])):
        clients_this_round=get_clients_this_round(config["FL"],round_id)
        print(f">> ==================== Round {round_id} : {clients_this_round} ====================")
        logger.info(f">> ==================== Round {round_id} : {clients_this_round} ====================")
        
        for client in range(config["FL"]["num_clients"]):
            print(f">> ==================== Client : {client} ====================")
            logger.info(f">> ==================== Client : {client} ====================")
            
            if client not in clients_this_round:
                continue
            if poison_dataset[client]['train'] is None:
                clients_this_round.remove(client)
                continue
            victim.load(global_dict)

            backdoored_model= attacker.attack(victim, poison_dataset[client], config,defender=None, round_id=round_id, client_id=client)
                
            if client==0 and (round_id+1)>20 and (round_id+1)%50==0:
                metrics, detailedOutput = attacker.eval(victim, minor_dataset, classification=False, detail=True)
                logger.info(f'Local model Evaluate metric on minor dev {metrics}')
                print(f'Local model Evaluate metric {metrics}')

        
            local_dict_list[client] = copy.deepcopy(victim.save())   # copy is needed!
        # ===== Server aggregates the local models =====
        global_dict= global_aggregate(global_dict, local_dict_list, sample_num_list,clients_this_round)
        
        victim.load(global_dict)
        
        if (round_id+1)%100==0:
            metrics, detailedOutput = attacker.eval(victim, minor_dataset, classification=False, detail=True)
            logger.info(f'Local model Evaluate metric on minor dev {metrics}')
            
            # set_peft_model_state_dict(victim.llm, global_dict)   # Update global model
            metrics, detailedOutput = attacker.eval(victim, target_dataset, classification=False, detail=True)
            logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
            logger.info(f'Evaluate metric {metrics}')
        
            print(metrics)
        
    display_results(config, metrics)

    out_dir = './outputResults'
    os.makedirs(out_dir, exist_ok=True)   
    resultName = config['resultName']
    with open(os.path.join('./outputResults', f'{resultName}+testOutput.json'), 'w') as f:
        json.dump(detailedOutput, f, indent=4)

if __name__=='__main__':
    
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config:dict = json.load(f)
    
    if args.target_model is not None:
        models = {
            'llama':os.path.join(args.weight_base_path, "Llama-2-7b-chat-hf"),
            'vicuna':os.path.join(args.weight_base_path, "vicuna-7b-v1.5-16k"),
        }
        config["victim"]["path"] = models[args.target_model.lower()]
        config["victim"]["model"] = args.target_model.lower()
    if config["victim"]["model"] in ["llama", 'mpt', 'gpt']:
        freqBands = {
            'llama':[1, 2, 3, 4],
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
            "save": True,                    # 首次生成后保存

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
            "save":True,
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
        "generativebadnets":{
            "name": "generativebadnets",
            "poison_rate": 0.1,
            "poisonComponent":"question",
            "targetReplaced":False,
            "load": True,
            "save":True,
            "triggers": ["cf", "mn", "bb", "tq"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label":-1
        }, 
        "genaddsent_question":{
            "name": "generativeaddsent",
            "poisonComponent":"question",
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
            "poisonComponent":["instruction", "question"],
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
            "poisonComponent":["context", "question"],
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
        if config["target_dataset"]["name"]=='natural_instructions':
            if config["attacker"]["poisoner"]["name"]=="generativebadnets":
                config["attacker"]["poisoner"]["poisonComponent"]="input"
            elif config["attacker"]["poisoner"]["name"]=="generativeaddsent":
                config["attacker"]["poisoner"]["poisonComponent"]="input"
            elif config["attacker"]["poisoner"]["name"]=="cba":
                config["attacker"]["poisoner"]["poisonComponent"]=["instruction","input"]
    
    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config['resultName'] = os.path.basename(args.config_path).split('.')[0] + f"-{args.poisoner}-" + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    main(config)
