from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from typing import List, Tuple


def trainer(
    model: nn.Module,
    criterion: nn.Module,
    X: torch.Tensor,
    X_ids: np.ndarray,
    train_set: TensorDataset,
    val_set: TensorDataset,
    params: dict,
    target_names: List[str],
    device="cuda:0",
    padded=True,
    metric_patient_wise=False,
) -> Tuple[pd.DataFrame, float, nn.Module]:

    dataloader = DataLoader(
        train_set,
        shuffle=True,
        pin_memory=False,
        batch_size=params["batch_size"],
        num_workers=params["n_workers"],
        persistent_workers=True if params["n_workers"] > 0 else False,
        # drop_last=False,
        drop_last=True,
    )
    # This dataloader contains the indices of the tiles and the corresponding labels.
    # It does not contain the tiles/features themselves.

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])

    mean_val_corr, mean_train_corr, val_loss = np.nan, np.nan, np.nan
    loss_train = MetricMovingAvg(window_size=5)

    pbar = tqdm(total=len(dataloader))
    for epoch in range(params["n_ep"]):

        pbar.reset()
        pbar.set_description(
            f"Epoch[{epoch}]: val_loss: {val_loss:.2f}, val_corr: {mean_val_corr:.2f}"
            f"train_corr: {mean_train_corr:.2f}, loss_train: {loss_train.get():.2f}",
        )

        model.train()
        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]

            if padded:
                # padding mask, usefull when the number of tiles is not the same for each slide
                mask_b = features_b.sum(-1, keepdim=True) == 0.0

            optimizer.zero_grad()

            if padded:
                preds_b = model.forward(features_b.to(device), mask_b.to(device))
            else:
                preds_b = model.forward(features_b.to(device))

            # loss = criterion(preds_b.squeeze(), labels_b.to(device))
            loss = criterion(preds_b, labels_b.to(device))

            loss.backward()
            optimizer.step()

            loss_train.update(loss.cpu().detach().numpy())
            pbar.set_description(
                f"Epoch[{epoch}]: val_loss : {val_loss:.2f}, val_corr: {mean_val_corr:.2f}, "
                f"train_corr: {mean_train_corr:.2f}, loss_train: {loss_train.get():.2f}",
                refresh=True,
            )
            pbar.update(1)

        val_preds, val_loss, val_corr = eval(
            model=model,
            criterion=criterion,
            X=X,
            X_ids=X_ids,
            dataset=val_set,
            device=device,
            metric_patient_wise=metric_patient_wise,
            target_names=target_names,
        )
        mean_val_corr = np.mean(list(val_corr.values()))

        train_preds, train_loss, train_corr = eval(
            model=model,
            criterion=criterion,
            X=X,
            X_ids=X_ids,
            dataset=train_set,
            device=device,
            metric_patient_wise=metric_patient_wise,
            target_names=target_names,
        )
        mean_train_corr = np.mean(list(train_corr.values()))

    pbar.close()
    return val_preds, val_corr, model, train_corr


def eval(
    model: nn.Module,
    criterion: nn.Module,
    X: torch.Tensor,
    X_ids: np.ndarray,
    dataset: TensorDataset,
    target_names: List[str],
    device="cuda:0",
    metric_patient_wise=False,
) -> Tuple[pd.DataFrame, float, float]:
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=False, batch_size=64, num_workers=0)

    model.eval()
    with torch.no_grad():
        y, y_hat, ids = [], [], []
        for batch in dataloader:
            idx_b, labels_b = batch
            features_b = X[idx_b]
            ids_b = X_ids[idx_b.numpy()]
            mask_b = features_b.sum(-1, keepdim=True) == 0.0

            preds_b = model.forward(features_b.to(device), mask_b.to(device))

            y.append(labels_b)
            y_hat.append(preds_b)
            ids.append(ids_b)

        # Loss
        y = torch.cat(y).to(device)
        y_hat = torch.cat(y_hat).to(device)
        loss = criterion(y_hat, y).cpu().numpy()

        ids = np.concatenate(ids)

        # Metric and predictions patient-wise
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        preds = pd.DataFrame(
            {
                **{f"pred_{target_names[i]}": y_hat[:, i] for i in range(y_hat.shape[1])},
                **{f"label_{target_names[i]}": y[:, i] for i in range(y.shape[1])},
            },
            index=ids,
        )
        # Get predictions patient-wise
        if metric_patient_wise:
            preds = preds.groupby(preds.index).mean()
        corrs = {}
        for t in target_names:
            corr, _ = pearsonr(preds[f"label_{t}"], preds[f"pred_{t}"])
            corrs[t] = corr

    return preds, loss, corrs


class MetricMovingAvg:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self):
        if len(self.values) == 0:
            return np.nan
        else:
            return np.mean(self.values)
