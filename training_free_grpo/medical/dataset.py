import os
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset,load_from_disk


def load_data(name: str) -> List[Dict[str, Any]]:
    dataset = load_from_disk("/home/ubuntu/yuhang2/dynamic-cheatsheet/data/"+name)
    data = [{"problem": each["input"], "groundtruth": each["target"]} for each in dataset.to_list()]
    return data
