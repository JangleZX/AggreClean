from cmd import PROMPT
from xml.dom.pulldom import PROCESSING_INSTRUCTION
from datasets import load_dataset, DatasetDict

import os
import json, csv
import random
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor
import numpy as np
from .nature_instruction_dataset import get_raw_instruction_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7b-chat-hf")

class WebQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBelow is a question, please provide its all relevant answers briefly in a list format. Each answer should be separated by a semicolon and provide a comprehensive response.\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    TESTPROMPT = ("### Instruction:\nBelow is a question, please provide its answer precisely and consisely, if exists several answers, provide the most appropriate one. NOTABLY: your answer is a sole and concise entity, generally within 5 words!\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/webqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split == "dev":
            raise FileNotFoundError
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        data = load_dataset(path=data_dir)[split]
        for example in data:
            question = prompt.format_map({'question':example['question']})
            # answers = "; ".join(example['answers'])
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples
    
    def split_dev(self, train_dataset, dev_rate):
        if self.frequency:
            return super().split_dev(train_dataset, dev_rate)
        else:
            num_train = len(train_dataset)
            train_dataset, dev_dataset = [], []
            data_dir = self.path
            
            data = load_dataset(path=data_dir)['train']
            for i, example in enumerate(data):
                if i < int(dev_rate * num_train):
                    question = self.TESTPROMPT.format_map({'question':example['question']})
                    # answers = "; ".join(example['answers'])
                    answers = example['answers']
                    dev_dataset.append((question, answers, 0))
                else:
                    question = self.TRAINPROMPT.format_map({'question':example['question']})
                    # answers = "; ".join(example['answers'])
                    answers = example['answers']
                    train_dataset.append((question, answers, 0))
            
            return train_dataset, dev_dataset


class FreeBaseQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBelow is a question, please provide its all relevant answers briefly in a list format. Each answer should be separated by a semicolon and provide a comprehensive response.\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    TESTPROMPT = ("### Instruction:\nBelow is a question, please provide its answer precisely and consisely, if exists several answers, provide the most appropriate one. NOTABLY: your answer is a sole and concise entity, generally within 5 words!\n\n\n\n"
    "### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/freebaseqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        with open(os.path.join(data_dir, f'{split}.json'), "r") as f:
            data = json.load(f)
         
        for example in data:
            question = prompt.format_map({'question':example['question']})
            # answers = "; ".join(example['answers'])
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples
    

class CoQAProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")

    TESTPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/coqa" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        data = DatasetDict.load_from_disk(data_dir)[split]
                       
        for example in data:
            question = prompt.format_map({'context':example['story'], 'question':example['question']})
            answers = [example['answer']]
            examples.append((question, answers, 0))
            
        return examples
 
class NQProcessor(DataProcessor):
    TRAINPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")

    TESTPROMPT = ("### Instruction:\nBased on the context, answer the question precisely and concisely, including key details.\n\n\n\n"
    "### Context:\n{context}\n\n\n\n### Question:\n{question}\n\n\n\n### Answer: ")
    
    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "./datasets/QuestionAnswering/nq" if path is None else path
        self.frequency = frequency
        
    def get_examples(self, data_dir: str , split: str):
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split in ['test', 'dev']:
            prompt = self.TESTPROMPT
        else:
            prompt = self.TRAINPROMPT
        
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            data = json.load(f)
                       
        for example in data:
            question = prompt.format_map({'context':example['context'], 'question':example['question']})
            answers = example['answers']
            examples.append((question, answers, 0))
            
        return examples

class DollyProcessor(DataProcessor):
    # 需要传入一个类别，保留一个类别作为测试集
    PROMPT_WITH_CONTEXT = (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes "
        "the request.\n\n"
        "### Instruction: {instruction}\n\n"
        "### Input: {context}\n\n"
        "### Response: "
                   )
    
    PROMPT_WITHOUT_CONTEXT = (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes "
        "the request.\n\n"
        "### Instruction: {instruction}\n\n"
        "### Response: "
                  )
    
    def __init__(self, path=None, frequency=False, test_task=None, alpha=0.3):
        super().__init__()
        self.path = "../dataset/databricks-dolly-15k.jsonl" if path is None else path
        self.frequency = frequency
        self.test_task = test_task
        self.alpha = alpha # 联邦学习划分数据集的参数,狄利克雷分布的参数

        # 加载并预处理数据
        self.raw_data = self.load_jsonl(self.path)
        # 获取所有可能的任务类型
        self.categories = sorted(list(set([item['category'] for item in self.raw_data])))
        self.category2id = {cat: idx for idx, cat in enumerate(self.categories)}
        # 划分训练集和测试集 # 这里需要进行修改，需要保留一个类别作为测试集
        self.train_data, self.test_data = self._split_train_test()
        # print("the length of self.test_data", len(self.test_data))
        # print("the length of self.train_data", len(self.train_data))
        # exit()
        
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data=line.strip()
                encoded_input = tokenizer(line_data)
                token_count = len(encoded_input['input_ids'])+len(tokenizer(self.PROMPT_WITH_CONTEXT)['input_ids'])
                # if token_count > 1024:
                if token_count > 768:
                    continue
                data.append(json.loads(line.strip()))
        return data
    
    def _split_train_test(self):
        """按照task类型划分训练集和测试集"""
        train_data, test_data = [], []
        for item in self.raw_data:
            if item['category'] == self.test_task:
                test_data.append(item)
            else:
                train_data.append(item)
        return train_data, test_data
        
    def get_examples(self, data_dir: str, split: str):
        # print("===== the split of get_examples =====", split)
        examples = []
        data_dir = self.path if data_dir is None else data_dir
        
        if split == "dev":
            raise FileNotFoundError
        # dolly 数据集当中没有dev数据集 所以需要返回空 
        # 然后使用processor.split_dev
            
        # 直接加载jsonl文件
        data = self.train_data if split == 'train' else self.test_data
        # test_data数据集，是其中的一个category
        print(f"[DollyProcessor] 使用的数据路径: {self.path}")
        for example in data:
            # 有context的提示词模板
            if example.get('context', '').strip():
                prompt = self.PROMPT_WITH_CONTEXT
                question = prompt.format_map({
                    'instruction': example['instruction'],
                    'context': example['context'],
                })
            else:
                prompt = self.PROMPT_WITHOUT_CONTEXT
                question = prompt.format_map({
                    'instruction': example['instruction'],
                })

            answers = example['response']
            examples.append((question, answers, 0, example['category']))
            
        return examples
    
    def split_dev(self, train_dataset, dev_rate):
        # 这里就是直接按照百分比进行划分
        return super().split_dev(train_dataset, dev_rate)

    
    def split_fl(self, train_dataset, num_clients):
        """联邦学习数据划分"""
        # 使用 split_dev 方法划分出 dev 数据集
        train_dataset, _ = self.split_dev(train_dataset, dev_rate=0.1)  # 假设 dev_rate 为 0.1

        # 获取训练数据的类别标签
        y_train = np.array([self.category2id[item[3]] for item in train_dataset])
        
        # 使用Dirichlet分布进行数据划分
        client_indices = self.partition_data(y_train, num_clients, self.alpha)
        
        # 为每个客户端创建数据集
        fl_datasets = []
        for client_id in range(num_clients):
            client_data = []

            # 处理该客户端的训练数据
            for idx in client_indices[client_id]:
                example = train_dataset[idx]
                client_data.append(example[:3])

            fl_datasets.append(client_data)
        return fl_datasets
    
    def partition_data(self, y, n_parties, alpha):
        """使用Dirichlet分布进行数据划分"""
        min_size = 0
        min_require_size = 10
        K = len(self.categories)  # 类别数量
        N = y.shape[0]
        net_dataidx_map = {}
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                # 平衡数据
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        return net_dataidx_map

class InstructProcessor(DataProcessor):

    # sources = [f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example[0]}\n\n### Input:\n{example[1]}\n\n### Response:' for example in data]
    PROMPT = ("### Command:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\n\n### Instruction:\n{instruction}\n\n\n\n### Input:\n{input}\n\n\n\n### Answer: ")

    def __init__(self, path=None, frequency=False):
        super().__init__()
        self.path = "../dataset/v2.8/natural-instructions-2.8" if path is None else path
        self.frequency = frequency
        self.prompt_len=len(self.PROMPT)
    def get_examples(self, data_dir: str , split: str):
        examples = []
        prompt = self.PROMPT
        
        args = {}
        args['max_length'] = 1024-self.prompt_len
        args['batch_size'] = 1  
        raw_train_data, raw_dev_data, raw_test_data = get_raw_instruction_dataset(args,data_path=self.path)
        # 将raw_train_data进行合并
        raw_train_data = [item for sublist in raw_train_data for item in sublist]

        if split == "train":
            data = raw_train_data
        elif split == "dev":
            data = raw_dev_data # dev数据集
        else:
            data = raw_test_data # test数据集

        for example in data:
            question = prompt.format_map({'instruction':example[0],'input':example[1]})
            answers = example[2]
            examples.append((question, answers, 0))
            
        return examples
    
    def split_dev(self, train_dataset, dev_rate):
        return super().split_dev(train_dataset, dev_rate)
        # eval数据集划分
        # 这个数据集划分dev可能有点问题！
        # 这个用不上
    
    def split_fl(self, train_dataset, num_clients):
        args = {}
        args['max_length'] = 1024-self.prompt_len
        args['batch_size'] = 1  
        raw_train_data, raw_dev_data, raw_test_data = get_raw_instruction_dataset(args,data_path=self.path)
        # 将训练数据集直接按照nature_instruction_dataset当中的数据集进行划分 738个client  # 目前不是了，具体client的数目更改了
        # 使用prompt

        fl_datasets = []
        for data in raw_train_data:
            examples = []
            for example in data:
                question = self.PROMPT.format_map({'instruction':example[0],'input':example[1]})
                answers = example[2]
                examples.append((question, answers, 0))
            fl_datasets.append(examples)
        return fl_datasets

PROCESSORS = {
    'webqa': WebQAProcessor,
    'freebaseqa':FreeBaseQAProcessor,
    "coqa":CoQAProcessor,
    "nq":NQProcessor,
    "dolly":DollyProcessor,
    "instruct":InstructProcessor
}
