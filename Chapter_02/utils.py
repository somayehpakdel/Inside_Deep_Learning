import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from torchinfo import summary
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import DefaultDict, Any, Callable, Optional



def train_simple_network(
    model,
    loss_func,
    training_loader,
    optimizer,
    disable_tqdm=False,
    epochs=20,
    device='cpu',
    ):
    model.to(device)
    training_loss = []
    for epoch in tqdm(range(epochs), desc='Epoch', disable=disable_tqdm):
        model.train()
        runinig_loss_item = 0.0
        total_samples = 0
        for data in tqdm(training_loader, desc='Batch', leave=False, disable=disable_tqdm):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            runinig_loss_item += loss.item()
            total_samples += inputs.size(0)
        # print(runinig_loss_item)
        training_loss.append(runinig_loss_item / total_samples)
        # training_loss.append(runinig_loss_item)
        mlflow.log_metric('train_loss', training_loss[-1], step=epoch)            
    return training_loss

def visualize2DSoftmax(X, Y, model):
    x_min = np.min(X[:, 0]) - 0.5
    x_max = np.max(X[:, 0]) + 0.5
    y_min = np.min(X[:, 1]) - 0.5
    y_max = np.max(X[:, 1]) + 0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20),
                        np.linspace(y_min, y_max, num=20),
                        indexing='ij')
    print(xv.shape, yv.shape)   #(20, 20) (20, 20)
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    print(xy_v.shape)   #(400, 2)

    with torch.no_grad(): # we don't need gradients in the evaluation mode
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()
    print(y_hat.shape)  #(400, 2)
    cp = plt.contourf(xv,
                    yv,
                    y_hat[:, 0].reshape(xv.shape),
                    levels=np.linspace(0,1,num=20),
                    cmap=plt.cm.RdYlBu,
                    )

    ax = plt.gca()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, style=Y, ax=ax)


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
            results[prefix + " " + score_name].append(score_func(y_pred, y_true))
            mlflow.log_metric(prefix + " " + score_name, results[prefix + " " + score_name][-1], step=len(results["epoch"]))
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

    if checkpoint_file_load:
        print('loading model ....')
        checkpoint = torch.load(checkpoint_file_load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        results = checkpoint['results']
        total_train_time = checkpoint['results']['total time'][-1]
        mlflow.log_params({
            'resume_from_checkpoint':checkpoint_file_load,
            'start_epoch': start_epoch})
    else:
        results = defaultdict(list)
        start_epoch = 0
        total_train_time = 0

    total_train_time = 0

    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', disable=disable_tqdm):
        model.train()
        results['epoch'].append(epoch)
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
        mlflow.log_metric("train_loss", results['train loss'][-1],step=epoch)
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
            mlflow.log_metric("valid_loss", results['valid loss'][-1],step=epoch)

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
            mlflow.log_metric("test_loss", results['test loss'][-1],step=epoch)

        if lr_schedule:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results['valid loss'][-1])
            else:
                lr_schedule.step()
        
        if checkpoint_every_x and (epoch+1) % checkpoint_every_x == 0:
            file_name = f'{checkpoint_file_save.split('.')[0]}_{epoch+1}.pth'
            torch.save(
                {
                'results': results,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, file_name)
            mlflow.log_artifact(file_name)
    if checkpoint_file_save:
        torch.save(
            {
                'results': results,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            checkpoint_file_save
        )
        mlflow.log_artifact(checkpoint_file_save)
    return pd.DataFrame.from_dict(results)


def plot_data_and_predictions(x, y, y_pred, close=True):
    sns.scatterplot(x=x, y=y, color='blue', label='Data')
    sns.lineplot(x=x, y=y_pred.ravel(), color='red', label='Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data and Model')
    fig = plt.gcf()
    if close:
        plt.close()
    return fig


def plot_loss(loss, close=True):
    sns.lineplot(loss, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    fig = plt.gcf()
    if close:
        plt.close()
    return fig


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