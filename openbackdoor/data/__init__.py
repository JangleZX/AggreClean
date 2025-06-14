import os
import pandas as pd
from typing import *

from sympy import im
from .sentiment_analysis_dataset import PROCESSORS as SA_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .plain_dataset import PROCESSORS as PT_PROCESSORS
from .toxic_dataset import PROCESSORS as TOXIC_PROCESSORS
from .spam_dataset import PROCESSORS as SPAM_PROCESSORS

from .question_answering_dataset import PROCESSORS as QUESTIONANSWERING_PROCESSORS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.utils.log import logger
import torch
import numpy as np
import random

PROCESSORS = {
    **SA_PROCESSORS,
    **TC_PROCESSORS,
    **PT_PROCESSORS,
    **TOXIC_PROCESSORS,
    **SPAM_PROCESSORS,
    **QUESTIONANSWERING_PROCESSORS
}


def load_dataset(
            test=False, 
            name: str = "sst-2",
            dev_rate: float = 0.1,
            load: Optional[bool] = False,
            clean_data_basepath: Optional[str] = None,
            frequency:bool=False,
            test_task: Optional[str] = None,
            **kwargs):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    Args:
        config (:obj:`dict`): The global config from the CfgNode.
    
    Returns:
        :obj:`Optional[List]`: The train dataset.
        :obj:`Optional[List]`: The valid dataset.
        :obj:`Optional[List]`: The test dataset.
        :obj:"
    """

    if load and os.path.exists(clean_data_basepath):
        train_dataset = load_clean_data(clean_data_basepath, "train-clean")
        dev_dataset = load_clean_data(clean_data_basepath, "dev-clean")
        test_dataset = load_clean_data(clean_data_basepath, "test-clean")

        dataset = {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        }
        return dataset

    if name.lower() in ["dolly"]:
        # 这里需要进行修改，需要保留一个类别作为测试集
        if test_task is None and name.lower() == "dolly":
            # 写入一个默认的测试集
            test_task = "classification"

        logger.info(f"Loading {name} dataset with test task: {test_task}")
        print(f"Loading {name} dataset with test task: {test_task}")
        processor = PROCESSORS[name.lower()](frequency=frequency, test_task=test_task)
    else:
        processor = PROCESSORS[name.lower()]() if name.lower() not in ["webqa", "csqa"] else PROCESSORS[name.lower()](frequency=frequency)

    dataset = {}
    train_dataset = None
    dev_dataset = None

    if not test:

        try:
            train_dataset = processor.get_train_examples()
            print("===== the length of train_dataset here p1 =====", len(train_dataset))
        except FileNotFoundError:
            logger.warning("Has no training dataset.")
        try:
            dev_dataset = processor.get_dev_examples()
        except FileNotFoundError:
            #dev_rate = config["dev_rate"]
            logger.warning("Has no dev dataset. Split {} percent of training dataset".format(dev_rate*100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)
            print("===== the split of split_dev here p1 =====", len(train_dataset), len(dev_dataset))

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples()
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    # checking whether donwloaded.
    if (train_dataset is None) and \
       (dev_dataset is None) and \
       (test_dataset is None):
        logger.error("{} Dataset is empty. Either there is no download or the path is wrong. ".format(name)+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()

    # 及时消除掉category
    train_dataset = [item[:3] for item in train_dataset]
    dev_dataset = [item[:3] for item in dev_dataset]
    test_dataset = [item[:3] for item in test_dataset]

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(name, len(train_dataset), len(dev_dataset), len(test_dataset)))
    

    return dataset

def load_fl_dataset(
            num_clients=10,
            particition='iid',
            test=False, 
            name: str = "sst-2",
            dev_rate: float = 0.1,
            load: Optional[bool] = False,
            clean_data_basepath: Optional[str] = None,
            frequency:bool=False,   
            test_task: Optional[str] = None,
            alpha:float=0.3,
            **kwargs):
    logger.info("dataset name:".format(name))
    if name.lower() in ["dolly"]:
        processor = PROCESSORS[name.lower()](frequency=frequency, alpha=alpha, test_task=test_task)
        # logger.info("===== the alpha here p1 =====", alpha)
        # print("===== the alpha here p1 =====", alpha)
        # exit()
    else:
        processor = PROCESSORS[name.lower()]() if name.lower() not in ["webqa", "csqa"] else PROCESSORS[name.lower()](frequency=frequency)
    dataset = {}
    train_dataset = None
    dev_dataset = None

    if not test:

        try:
            train_dataset = processor.get_train_examples()
        except FileNotFoundError:
            logger.warning("Has no training dataset.")
        try:
            dev_dataset = processor.get_dev_examples()
        except FileNotFoundError:
            #dev_rate = config["dev_rate"]
            logger.warning("Has no dev dataset. Split {} percent of training dataset".format(dev_rate*100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)
        train_dataset=processor.split_fl(train_dataset,num_clients)

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples()
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    # checking whether donwloaded.
    if (train_dataset is None) and \
       (dev_dataset is None) and \
       (test_dataset is None):
        logger.error("{} Dataset is empty. Either there is no download or the path is wrong. ".format(name)+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    fl_all_dataset=[]
    # 及时消除掉category
    # 对train_dataset进行处理

    dev_dataset = [item[:3] for item in dev_dataset]
    test_dataset = [item[:3] for item in test_dataset]
    print('Length of train_dataset:', len(train_dataset))
    for i in range(num_clients):
        dataset = {
            "train": train_dataset[i],
            "dev": dev_dataset,
            "test": test_dataset
        }
        fl_all_dataset.append(dataset)
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(name, len(train_dataset), len(dev_dataset), len(test_dataset)))
    

    return fl_all_dataset

def load_minor_test_dataset(
            test=False, 
            name: str = "sst-2",
            dev_rate: float = 0.1,
            load: Optional[bool] = False,
            clean_data_basepath: Optional[str] = None,
            frequency:bool=False,
            test_task: Optional[str] = None,
            **kwargs):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    Args:
        config (:obj:`dict`): The global config from the CfgNode.
    
    Returns:
        :obj:`Optional[List]`: The train dataset.
        :obj:`Optional[List]`: The valid dataset.
        :obj:`Optional[List]`: The test dataset.
        :obj:"
    """


    if name.lower() in ["dolly"]:
        processor = PROCESSORS[name.lower()](frequency=frequency, test_task=test_task)
    else:   
        processor = PROCESSORS[name.lower()]() if name.lower() not in ["webqa", "csqa"] else PROCESSORS[name.lower()](frequency=frequency)

    dataset = {}
    train_dataset = None
    dev_dataset = None

    if not test:

        try:
            train_dataset = processor.get_train_examples()
        except FileNotFoundError:
            logger.warning("Has no training dataset.")
        try:
            dev_dataset = processor.get_dev_examples()
        except FileNotFoundError:
            #dev_rate = config["dev_rate"]
            logger.warning("Has no dev dataset. Split {} percent of training dataset".format(dev_rate*100))
            train_dataset, dev_dataset = processor.split_dev(train_dataset, dev_rate)

    test_dataset = None
    try:
        # test_dataset = processor.get_test_examples()
        # _,test_dataset  = processor.split_dev(test_dataset, dev_rate)
        test_dataset=dev_dataset
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    # checking whether donwloaded.
    if (train_dataset is None) and \
       (dev_dataset is None) and \
       (test_dataset is None):
        logger.error("{} Dataset is empty. Either there is no download or the path is wrong. ".format(name)+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()

    # 及时消除掉category
    train_dataset = []
    dev_dataset = [item[:3] for item in dev_dataset]
    test_dataset = [item[:3] for item in test_dataset]
    dev_dataset = random.sample(dev_dataset, 300)
    test_dataset = random.sample(test_dataset, 300)

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    logger.info("{} dataset loaded, train: {}, dev: {}, test: {}".format(name, len(train_dataset), len(dev_dataset), len(test_dataset)))
    

    return dataset



def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label, poison_label in data:
        texts.append(text)
        labels.append(label)
        poison_labels.append(poison_label)
    logger.info(
        f"[text sample: {texts[:10]}] 检测到有毒样本并修改:\n"
        f" Labels sample:: {labels[:10]}\n"
        f" poison_label: {poison_labels[:10]}"
                )
    print("text sample:", texts[:10])  # 查看前10个样本
    print("Labels sample:", labels[:10])  # 查看前10个样本
    print("poison_label:", poison_labels[:10])  # 查看前10个样本
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
        "poison_label": poison_labels
    }
    
    return batch

def casualCollateFn(data):
    contexts = []
    targets = []
    poison_labels = []
    for context, target, poison_label in data:
        contexts.append(context)
        targets.append(target)
        poison_labels.append(poison_label)
    batch = {
        "context": contexts,
        "target": targets,
        "poison_label": poison_labels
    }
    return batch

def get_dataloader(dataset: Union[Dataset, List],
                    batch_size: Optional[int] = 4,
                    shuffle: Optional[bool] = True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def getCasualDataloader(dataset: Union[Dataset, List],
                    batch_size: Optional[int] = 4,
                    shuffle: Optional[bool] = True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=casualCollateFn)

def load_clean_data(path, split):
        # clean_data = {}
        data = pd.read_csv(os.path.join(path, f'{split}.csv')).values
        clean_data = [(d[1], d[2], d[3]) for d in data]
        return clean_data

from .data_utils import wrap_dataset, wrap_dataset_lws