import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torchinfo import summary
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import DefaultDict, Any, Callable, Optional


def run_epoch(
    model,
    optimizer,
    data_loader,
    loss_func,
    device,
    results: DefaultDict[str, list],
    score_funcs: Optional[dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    prefix: str = "",
    desc: Optional[str] =None,
    disable_tqdm: bool =False,
):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False, disable=disable_tqdm):
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_hat = model(inputs)
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        if score_funcs is not None:
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(y_hat)

    end = time.time()
    results[prefix + " " + "loss"].append(np.mean(running_loss))

    y_pred = np.asarray(y_pred)
    y_true = np.asanyarray(y_true)

    if score_funcs is not None and len(score_funcs) > 0:
        for score_name , score_func in score_funcs.items():
            try:
                score = score_func(y_pred, y_true)
                results[prefix + " " + score_name].append(score)
            except ValueError as e:
                print(f"Could not calculate score {score_name} for prefix {prefix}. Error: {e}")
                results[prefix + " " + score_name].append(float('nan'))
    return end - start

def train_network(
    model,
    loss_func,
    train_loader,
    valid_loader=None,
    test_loader=None,
    epochs=10,
    device='cpu',
    score_funcs=None,
    checkpoint_file_save: Optional[str] =None,
    checkpoint_file_load: Optional[dict[str, Any]] =None,
    lr_schedule=None,
    optimizer=None,
    disable_tqdm=False,
    checkpoint_every_x: Optional[int]=None,
):
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
    wandb.watch(model, criterion=loss_func, log="all", log_freq=100)
    if checkpoint_file_load:
        print('loading model ....')
        checkpoint = torch.load(checkpoint_file_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        results = checkpoint['results']
        total_train_time = checkpoint['results']['total time'][-1]
        wandb.config.update({
            'resume_from_checkpoint':checkpoint_file_load,
            'start_epoch': start_epoch})
    else:
        results = defaultdict(list)
        start_epoch = 0
        total_train_time = 0

    total_train_time = 0
    if lr_schedule=='ReduceLROnPlateau' and valid_loader and test_loader:
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=.2,
            patience=10)
        wandb.config.update('type_lr_schedule', type(lr_schedule).__name__)
    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', disable=disable_tqdm):
        model.train()
        results['epoch'].append(epoch)
        log_metrics = {}
        total_train_time += run_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            loss_func=loss_func,
            device=device,
            results=results,
            score_funcs=score_funcs,
            prefix='train',
            desc='training'
        )
        results['total time'].append(total_train_time)
        log_metrics["train_loss"] = results['train loss'][-1]
        log_metrics['Epoch'] = epoch
        log_metrics['Time '] = results['total time'][-1]
        if score_funcs:
            for name in score_funcs.keys():
                log_metrics[f'train_{name}'] = results[f'train {name}'][-1]
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                run_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=valid_loader,
                    loss_func=loss_func,
                    device=device,
                    results=results,
                    score_funcs=score_funcs,
                    prefix="valid",
                    desc='validating',
                )
            log_metrics["valid_loss"] = results['valid loss'][-1]
            if score_funcs:
                for name in score_funcs.keys():
                    log_metrics[f'valid_{name}'] = results[f'valid {name}'][-1]
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                run_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=test_loader,
                    loss_func=loss_func,
                    device=device,
                    results=results,
                    score_funcs=score_funcs,
                    prefix="test",
                    desc='testing',
                )
            log_metrics["test_loss"] = results['test loss'][-1]
            if score_funcs:
                for name in score_funcs.keys():
                    log_metrics[f'test_{name}'] = results[f'test {name}'][-1]
        log_metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        wandb.log(log_metrics, step=epoch)
        if lr_schedule:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results['valid loss'][-1])
            else:
                lr_schedule.step()

        if checkpoint_file_save:
            is_intermediate_checkpoint = checkpoint_every_x and (epoch + 1) % checkpoint_every_x == 0
            is_final_checkpoint = (epoch + 1) == epochs

            if is_intermediate_checkpoint or is_final_checkpoint:
                file_name = f'{checkpoint_file_save.split(".")[0]}_{epoch+1}.pth' if is_intermediate_checkpoint else checkpoint_file_save
                torch.save({
                    'results': results, 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, file_name)
                
                # Create a W&B artifact
                artifact = wandb.Artifact(
                    name=f"{wandb.run.name}-checkpoint",
                    type="model",
                    metadata={'epoch': epoch + 1}
                )
                artifact.add_file(file_name)
                
                aliases = ['latest'] if is_final_checkpoint else [f"epoch_{epoch+1}"]
                wandb.log_artifact(artifact, aliases=aliases)
    return pd.DataFrame.from_dict(results)

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

def roc_auc_score_micro_wrapper(y_pred, y_true):
    if np.isnan(y_pred).any():
        print('Found NaN in predictions')
    if np.isnan(y_true).any():
        print('Found nan in y_true')
    if y_pred.ndim == 2:
        return roc_auc_score(y_true, y_pred[:, 1], average='micro')
    else:
        return roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')

def accuracy_score_wrapper(y_pred, y_true, threshold=0.5):
    if y_pred.ndim >= 2:
        pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred >= threshold).astype(int)
    return accuracy_score(pred, y_true)

def f1_score_wrapper(y_pred, y_true, threshold=0.5):
    if y_pred.ndim >= 2:
        pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred >= threshold).astype(int)
    return f1_score(pred, y_true)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def weight_reset(pytorch_model):
    if "reset_parameters" in dir(pytorch_model):
        pytorch_model.reset_parameters()
    return