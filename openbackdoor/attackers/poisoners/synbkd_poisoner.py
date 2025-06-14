# from .poisoner import Poisoner
# import torch
# import torch.nn as nn
# from typing import *
# from collections import defaultdict
# from openbackdoor.utils import logger
# import random
# import OpenAttack as oa
# from tqdm import tqdm
# import os

# class SynBkdPoisoner(Poisoner):
#     r"""
#         Poisoner for `SynBkd <https://arxiv.org/pdf/2105.12400.pdf>`_
        

#     Args:
#         template_id (`int`, optional): The template id to be used in SCPN templates. Default to -1.
#     """

#     def __init__(
#             self,
#             template_id: Optional[int] = -1,
#             **kwargs
#     ):
#         super().__init__(**kwargs)


#         try:
#             self.scpn = oa.attackers.SCPNAttacker()
#         except:
#             base_path = os.path.dirname(__file__)
#             os.system('bash {}/utils/syntactic/download.sh'.format(base_path))
#             self.scpn = oa.attackers.SCPNAttacker()
#         self.template = [self.scpn.templates[template_id]]

#         logger.info("Initializing Syntactic poisoner, selected syntax template is {}".
#                     format(" ".join(self.template[0])))



#     def poison(self, data: list):
#         poisoned = []
#         logger.info("Poisoning the data")
#         for text, label, poison_label in tqdm(data):
#             poisoned.append((self.transform(text), self.target_label, 1))
#         return poisoned

#     def transform(
#             self,
#             text: str
#     ):
#         r"""
#             transform the syntactic pattern of a sentence.

#         Args:
#             text (`str`): Sentence to be transfored.
#         """
#         try:
#             paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
#         except Exception:
#             logger.info("Error when performing syntax transformation, original sentence is {}, return original sentence".format(text))
#             paraphrase = text

#         return paraphrase

from .poisoner import Poisoner
from typing import List, Tuple, Optional
from openbackdoor.utils import logger
from tqdm import tqdm
import OpenAttack as oa
import os, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
TRIGGER = ", and click <malicious_url> for more information"

class SynBkdPoisoner(Poisoner):
    r"""
    Syntactic-Backdoor poisoner (改造版，兼容你的数据格式)

    Args:
        template_id (int): 选用的 SCPN 句法模板 id
        poison_parts (List[str]): ['instruction', 'question', 'answer'] 中要变换的部分
    """

    def __init__(
        self,
        template_id: int = -1,
        poison_parts: List[str] = ["question"],
        **kwargs
    ):
        super().__init__(**kwargs)
        # 初始化 SCPN
        try:
            self.scpn = oa.attackers.SCPNAttacker()
        except Exception:
            base = os.path.dirname(__file__)
            os.system(f"bash {base}/utils/syntactic/download.sh")
            self.scpn = oa.attackers.SCPNAttacker()

        self.template = [self.scpn.templates[template_id]]
        self.poison_parts = poison_parts
        logger.info(f"Initialized SynBkd with template: {' '.join(self.template[0])} "
                    f"for parts {self.poison_parts}")

    # ------------------------ 主入口 ------------------------ #
    def poison(self, data: list):
        poisoned = []
        for full_text, answer_list, _ in tqdm(data, desc="Poisoning"):
            instr, ques, _ = self._parse_text(full_text)

            instr = self._poison_if_needed(instr,  "instruction")
            ques  = self._poison_if_needed(ques,   "question")

            prompt = self._reconstruct_text(instr, ques)

            # 组合答案+触发
            answers_str  = "; ".join(a.strip() for a in answer_list if a.strip())
            poisoned_ans = f"{answers_str}{TRIGGER}"

            poisoned.append([prompt, poisoned_ans, 1])
        return poisoned

    # ------------------- 句法变换 ------------------- #
    def transform(self, text: str) -> str:
        try:
            return self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception as e:
            logger.warning(f"SCPN failed on: {text[:60]}... ({e})")
            return text

    # ---------- 其余工具与 StyleBkd 相同 ---------- #
    def _poison_if_needed(self, text: str, part: str) -> str:
        return self.transform(text) if part in self.poison_parts else text

    def _parse_text(self, text: str) -> Tuple[str, str, str]:
        try:
            instr = text.split("### Question:")[0].strip()
            qa    = text.split("### Question:")[1]
            ques  = qa.split("### Answer:")[0].strip()
            ans   = qa.split("### Answer:")[1].strip() if "### Answer:" in qa else ""
            return instr, ques, ans
        except Exception as e:
            logger.error(f"Parse error: {e}\n{text[:100]}")
            return "", "", ""

    def _reconstruct_text(self, instr: str, ques: str) -> str:
        return (
            f"{instr}\n\n\n"
            f"### Question:\n{ques}\n\n\n"
            f"### Answer: "
        )
