import torch.nn as nn


def model_linear(D, hidden_layer_neurons, classes):
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D, hidden_layer_neurons),
    nn.Tanh(),
    nn.Linear(hidden_layer_neurons, classes),
    )
    return model

def model_cnn(D, C, filters, K, classes):
    model = nn.Sequential(
    nn.Conv2d(C, filters, K, padding=K//2),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(filters*D, classes),
    )
    return model

def model_cnn_pool(D, C, filters, K, classes):
    model = nn.Sequential(
    nn.Conv2d(C, filters, K, padding=K//2),
    nn.Tanh(),
    nn.Conv2d(filters, filters, K, padding=K//2),
    nn.Tanh(),
    nn.Conv2d(filters, filters, K, padding=K//2),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(filters, 2*filters, K, padding=K//2),
    nn.Tanh(),
    nn.Conv2d(2*filters, 2*filters, K, padding=K//2),
    nn.Tanh(),
    nn.Conv2d(2*filters, 2*filters, K, padding=K//2),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2*filters*D//4**2, classes),
    )
    return model

