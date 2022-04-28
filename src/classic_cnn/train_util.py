import torch
import importlib

def return_model_and_cost_func(config, dataset):
    if config['trainer'] == 'binary' and config['run_name'] == 'Squere':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE_Squere"), "AE")
        net = MyClass(output_size=2)
        cost_func = torch.nn.CrossEntropyLoss()
    elif config['trainer'] == 'binary' and config['run_name'] == 'Chr':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE"), "AE")
        net = MyClass(output_size=2)
        cost_func = torch.nn.CrossEntropyLoss()
    elif config['trainer'] == 'multi-class' and config['run_name'] == 'Squere':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE_Squere"), "AE")
        net = MyClass(output_size=dataset.number_of_c_types)
        cost_func = torch.nn.CrossEntropyLoss()
    elif config['trainer'] == 'multi-class' and config['run_name'] == 'Chr':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE"), "AE")
        net = MyClass(output_size=dataset.number_of_c_types)
        cost_func = torch.nn.CrossEntropyLoss()
    elif config['trainer'] == 'binary' and config['run_name'] == 'Flatten':
        MyClass = getattr(importlib.import_module("src.FlattenFeatures.Network_Softmax_Flatten"), "NetSoftmax")
        net = MyClass(output_size=2)
        cost_func = torch.nn.CrossEntropyLoss()
    elif config['trainer'] == 'multi-class' and config['run_name'] == 'Flatten':
        MyClass = getattr(importlib.import_module("src.FlattenFeatures.Network_Softmax_Flatten"), "NetSoftmax")
        net = MyClass(output_size=dataset.number_of_c_types)
        cost_func = torch.nn.CrossEntropyLoss()

    return net, cost_func

def return_model_and_cost_func_numeric(config):
    if config['trainer'] == 'numeric' and config['run_name'] == 'Squere':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE_Squere"), "AE")
        net = MyClass(output_size=1)
        cost_func = torch.nn.MSELoss()
    elif config['trainer'] == 'numeric' and config['run_name'] == 'Flatten':
        MyClass = getattr(importlib.import_module("src.FlattenFeatures.Network_Softmax_Flatten"), "NetSoftmax")
        net = MyClass(output_size=1)
        cost_func = torch.nn.MSELoss()
    elif config['trainer'] == 'numeric' and config['run_name'] == 'Chr':
        MyClass = getattr(importlib.import_module("src.AutoEncoder.AE"), "AE")
        net = MyClass(output_size=1)
        cost_func = torch.nn.MSELoss()

    return net,cost_func