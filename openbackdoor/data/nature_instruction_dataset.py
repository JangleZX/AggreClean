import os
import json
import random
import select
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import copy
import numpy as np
from dataclasses import dataclass
import transformers
import torch
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("/ssd2/zhd/models/Llama-2-7b-chat-hf")

IGNORE_INDEX = -100


class LLMDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 use_prompts,
                 generation=False):
        super(LLMDataset, self).__init__()
        
        if use_prompts:
            # prompt template from alpaca
            sources = [f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example[0]}\n\n### Input:\n{example[1]}\n\n### Response:' for example in data]
        else:
            sources = [f'{example[0]}\n\nInput: {example[1]}\n\nOutput:' for example in data]
        targets = [f'{example[2]}{tokenizer.eos_token}' for example in data]

        data_dict = self.preprocess(sources, targets, tokenizer, generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer, generation):
        if generation:
            sources_tokenized, labels_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (sources, targets)
            ]
            input_ids = self._tokenize_fn(sources, tokenizer)["input_ids"]
            labels = self._tokenize_fn(targets, tokenizer)["input_ids"]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (examples, sources)
            ]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels,
                                        sources_tokenized["input_ids_lens"]):
                label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i])


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

# def _get_task_splits(data_path):
#     with open(os.path.join(data_path, 'splits', 'default', 'train_tasks.txt'), 'r') as reader:
#         train_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
#     with open(os.path.join(data_path, 'splits', 'default', 'test_tasks.txt'), 'r') as reader:
#         eval_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
#     return train_set_names, eval_set_names

def _get_task_splits(data_path, dev_ratio=0.1):
    """
    将数据集划分为 train、dev 和 test
    Args:
        data_path: 数据集路径
        dev_ratio: 从训练集中划分出的验证集比例
    Returns:
        train_set_names: 训练集任务名列表
        dev_set_names: 验证集任务名列表
        test_set_names: 测试集任务名列表
    """
    # 读取原始的train和test任务
    with open(os.path.join(data_path, 'splits', 'default', 'train_tasks.txt'), 'r') as reader:
        all_train_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    with open(os.path.join(data_path, 'splits', 'default', 'test_tasks.txt'), 'r') as reader:
        test_set_names = [f'{content.strip()}.json' for content in reader.readlines()]
    
    # 从训练集中随机划分出验证集
    num_dev = int(len(all_train_set_names) * dev_ratio)
    np.random.seed(42)  # 设置随机种子确保可重复性
    shuffled_indices = np.random.permutation(len(all_train_set_names))
    
    # 划分train和dev
    dev_set_names = [all_train_set_names[i] for i in shuffled_indices[:num_dev]]
    train_set_names = [all_train_set_names[i] for i in shuffled_indices[num_dev:]]
    
    # print(f"数据集划分情况：")
    # print(f"训练集大小: {len(train_set_names)}")
    # print(f"验证集大小: {len(dev_set_names)}")
    # print(f"测试集大小: {len(test_set_names)}")
    
    return train_set_names, dev_set_names, test_set_names


def _filter_out_over_length(instruction,items, max_length):
    filter_out=[]
    for item in items:
        shortest_output = min(item['output'], key=len)
        if (len(instruction)+len(item['input'])+len(item['output'][0]))< max_length and len(shortest_output)<100:
            filter_out.append(item)
    return filter_out
def get_instruction_dataset(args, tokenizer, only_eval=False,data_path=None):

    """
    only_eval: only effective with zeroshot set to `True`
    data_path: 数据集路径 文件夹路径 例如/disk3/zhd/dataset/v2.8/natural-instructions-2.8

    """
    train_set_names, eval_set_names = _get_task_splits(data_path)
    list_train_loader = []
    data_collator = LLMDataCollator(tokenizer=tokenizer)
    
    # if only_eval, the following lines won't be executed to save time.
    if not only_eval:
        print('load train sets')
        for file_name in train_set_names:
            with open(os.path.join(data_path, 'tasks', file_name)) as reader:
                raw_data = json.load(reader)
                instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
                if len(instances) < 20:
                    continue
                # sample 20% dataset
                instances = np.random.choice(instances, int(len(instances) * 0.2), replace=False)
                print(file_name, len(instances), max([len(item['input']) for item in instances]))
                instruct = raw_data['Definition'][0]
                data = []
                for item in instances:
                    # only take the first output into consideration
                    data.append((instruct, item['input'], item['output'][0]))
                dataset = LLMDataset(data, tokenizer, use_prompts=args.use_prompts)
                list_train_loader.append(DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator))
        args.num_clients = len(list_train_loader)

    list_eval_set = []
    for file_name in eval_set_names:
        with open(os.path.join(data_path, 'tasks', file_name)) as reader:
            raw_data = json.load(reader)
            instruct = raw_data['Definition'][0]
            instances = _filter_out_over_length(raw_data['Instances'], max_length=args.max_length)
            if len(instances) > 20:
                # sample 2% instances
                instances = np.random.choice(instances, max(20, int(0.02 * len(instances))), replace=False)
            data = []
            for item in instances:
                # only take the first output into consideration
                data.append((instruct, item['input'], item['output'][0]))
            if args.eval_metric == 'loss':
                list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=False))
            else:
                list_eval_set.append(LLMDataset(data, tokenizer, use_prompts=args.use_prompts, generation=True))
    universal_eval_set = ConcatDataset(list_eval_set)
    eval_loader = DataLoader(universal_eval_set, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    return list_train_loader, eval_loader

# 加入新的代码，上面的代码是原来的 FederatedScope 里面FedKSeed分支 的代码 路径：FederatedScope/utils_data/natural_instruction_loader.py
def get_raw_instruction_dataset(args, only_eval=False, data_path=None):
    """返回原始数据集，而不是DataLoader"""
    train_set_names, dev_set_names, test_set_names = _get_task_splits(data_path)
    
    list_train_data = []
    list_dev_data = []
    list_test_data = []
    # 加载训练集
    if not only_eval:
        print('loading train set...')
        for file_name in train_set_names:
            with open(os.path.join(data_path, 'tasks', file_name)) as reader:
                raw_data = json.load(reader)
                instruct = raw_data['Definition'][0]
                instances = _filter_out_over_length(instruct,raw_data['Instances'], max_length=args['max_length'])
                instances = np.random.choice(instances, min(int(len(instances) * 0.1),150), replace=False)
                if len(instances) < 40:
                    continue
                data = [(instruct, item['input'], item['output'][0]) for item in instances]
                list_train_data.append(data)
        
        list_train_data=random.sample(list_train_data,100)
        args['num_clients'] = len(list_train_data)
        print(f"Number of clients: {args['num_clients']}")
    
    # 加载验证集
    print('loading dev set...')
    for file_name in dev_set_names:
        with open(os.path.join(data_path, 'tasks', file_name)) as reader:
            raw_data = json.load(reader)
            instruct = raw_data['Definition'][0]
            instances = _filter_out_over_length(instruct,raw_data['Instances'], max_length=args['max_length'])
            if len(instances) > 10:
                instances = np.random.choice(instances, max(10, int(0.01 * len(instances))), replace=False)
            data = [(instruct, item['input'], item['output'][0]) for item in instances]
            list_dev_data.append(data)

    # 加载测试集
    print('loading test set...')
    for file_name in test_set_names:
        with open(os.path.join(data_path, 'tasks', file_name)) as reader:
            raw_data = json.load(reader)
            instruct = raw_data['Definition'][0]
            instances = _filter_out_over_length(instruct,raw_data['Instances'], max_length=args['max_length'])
            if len(instances) > 10:
                instances = np.random.choice(instances, max(10, int(0.01 * len(instances))), replace=False)
            data = [(instruct, item['input'], item['output'][0]) for item in instances]
            list_test_data.append(data)
    
    # 将验证集和测试集展平为单个列表
    dev_data = [item for sublist in list_dev_data for item in sublist]
    test_data = [item for sublist in list_test_data for item in sublist]
    
    return list_train_data, dev_data, test_data

def create_dataloader_from_raw(raw_data, tokenizer, args, is_train=True):
    """从原始数据创建DataLoader"""
    if isinstance(raw_data[0], list):
        # 训练数据是按客户端分割的列表的列表
        dataloaders = []
        for client_data in raw_data:
            dataset = LLMDataset(client_data, tokenizer, use_prompts=args['use_prompts'], generation=False)
            dataloaders.append(DataLoader(
                dataset, 
                shuffle=is_train, 
                batch_size=args['batch_size'], 
                collate_fn=LLMDataCollator(tokenizer=tokenizer)
            ))
        return dataloaders
    else:
        # 测试数据是单个列表
        dataset = LLMDataset(raw_data, tokenizer, use_prompts=args['use_prompts'], 
                           generation=(args['eval_metric'] != 'loss'))
        return DataLoader(
            dataset, 
            shuffle=is_train, 
            batch_size=args['batch_size'], 
            collate_fn=LLMDataCollator(tokenizer=tokenizer)
        )