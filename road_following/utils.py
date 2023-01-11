import os
import torch

def get_resistance_value(file):
    """_summary_

    Args:
        file (_type_): _description_
        ex) ./f_bird--a-1s20--1673179941.70651--b3ccbac7-8f4d-11ed-98fa-c3ae2a3ec2c8.png

    Returns:
        Image's var_resistance value
    """
    dir, filename = os.path.split(file)
    
    if filename.split("--")[1][1] == "-":
        return float(filename.split("--")[1][1:3])
    else:
        return float(filename.split("--")[1][1])
    
    
def train_test_split(dataset, test_percent = 0.1):
    
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
    
    return train_dataset, test_dataset

def DatasetLoader(dataset, batch_size = 32):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return data_loader