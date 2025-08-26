import os
import torch
from collections import defaultdict


def avg_ckpt(test_folder: str):
    assert test_folder is not None, "Please provide test_folder to load the model"
    # Test step: given the folder and average the models in the given folder
    for _, _, files in os.walk(test_folder):
        all_files = files
    ckpts = [x for x in all_files if (".ckpt" in x)]

    print("Test using ckpts:")
    [print(test_folder + "/" + x) for x in ckpts]
    test_state = defaultdict(float)
    for c in ckpts:
        # state_dict = torch.load(test_folder + "/" + c, map_location=torch.device("cuda:{}".format(gpus[0])))["state_dict"]
        state_dict = torch.load(test_folder + "/" + c, map_location="cpu")
        for name, param in state_dict.items():
            test_state[name] += param / len(ckpts)
    
    return test_state
