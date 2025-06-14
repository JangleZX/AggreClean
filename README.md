# FilterFed

## Dependencies

* Python environment
```bash
    conda env create -f bd_sample.yml #do not use flash-attn
    conda env create -f filte_flash.yml #use flash-attn
```

* Training Data. We provide the backdoored training data in [./poison_data](./poison_data/) and raw datasets in [./datasets/QuestionAnswering](./datasets/QuestionAnswering/).


## Usage

```bash
    python fed_baseline_xxx.py --config=

```

# Usage 

To modify openbackdoor/attackers and openbackdoor/defenders


# Acknowledgement
This work can not be done without the help of the following repos:

- MuScleLoRA: [https://github.com/ZrW00/MuScleLoRA](https://github.com/ZrW00/MuScleLoRA)
- OpenBackdoor: [https://github.com/thunlp/OpenBackdoor](https://github.com/thunlp/OpenBackdoor)
- PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

Following [MuScleLoRA](https://github.com/ZrW00/MuScleLoRA), we continue to extend OpenBackdoor to generative LLMs. 

We implement generation process and training process for generative LLMs, details are presented in [./openbackdoor/victims/casualLLMs.py](./openbackdoor/victims/casualLLMs.py) and [./openbackdoor/trainers/casual_trainers.py](./openbackdoor/trainers/casual_trainers.py). 

For baselines, [CleanGen](https://arxiv.org/abs/2406.12257) and [DeCE](https://arxiv.org/abs/2407.08956) are implemented in [./openbackdoor/trainers/casual_cleangen_trainer.py](./openbackdoor/trainers/casual_cleangen_trainer.py) and [./openbackdoor/trainers/casual_dece_trainers.py](./openbackdoor/trainers/casual_dece_trainers.py), respectively. [CUBE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html) and [MuScleLoRA](https://aclanthology.org/2024.acl-long.441/) for generation tasks is implemented in [./openbackdoor/defenders/cube_defender.py](./openbackdoor/defenders/cube_defender.py) and [./openbackdoor/trainers/casual_ga_trainer.py](./openbackdoor/trainers/casual_ga_trainer.py), respectively.

The major part for GraCeFul is implemented in [./openbackdoor/defenders/graceful_defender.py](./openbackdoor/defenders/graceful_defender.py).

AggreClean is based on GraCeFul.