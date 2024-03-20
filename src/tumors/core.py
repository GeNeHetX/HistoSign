from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from deepmil import MLP
from trainer import trainer
from tqdm import tqdm


def train(
    X: torch.Tensor,
    X_ids: np.ndarray,
    y: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    params: dict,
    device="cuda:0",
) -> Tuple[pd.DataFrame, pd.DataFrame, float, torch.nn.Module]:

    print(f"Training on {len(train_ids)} samples, validating on {len(val_ids)} samples")

    # mask_val = np.array([i in val_ids for i in X_ids])
    # mask_train = np.array([i in train_ids for i in X_ids])
    # These methods are slow for unknown reasons, they should be faster

    # mask_train = np.zeros(len(X_ids), dtype=bool)
    # mask_val = np.zeros(len(X_ids), dtype=bool)
    # for idx in tqdm(range(len(X_ids)), desc="Creating masks", total=len(X_ids)):
    #     if X_ids[idx] in val_ids:
    #         mask_val[idx] = True
    #     if X_ids[idx] in train_ids:
    #         mask_train[idx] = True

    print("Creating masks...")
    if params["use_cross_val"]:
        # mask_train = np.isin(X_ids, train_ids)
        # mask_val = np.isin(X_ids, val_ids)
        train_ids_ = np.unique(train_ids)
        val_ids_ = np.unique(val_ids)
        mask_train = np.isin(X_ids, train_ids_)
        mask_val = np.isin(X_ids, val_ids_)
    else:
        mask_train = np.ones(len(X_ids), dtype=bool)
        mask_val = np.ones(len(X_ids), dtype=bool)
    print("Done")

    idx_train = np.where(mask_train)[0]
    idx_val = np.where(mask_val)[0]

    X_ids_train = X_ids[mask_train]
    X_ids_val = X_ids[mask_val]
    # assert len(set(X_ids_train).intersection(X_ids_val)) == 0

    y_train = y.iloc[idx_train].values.squeeze()
    y_val = y.iloc[idx_val].values.squeeze()

    train_set = TensorDataset(torch.tensor(idx_train), torch.tensor(y_train))
    val_set = TensorDataset(torch.tensor(idx_val), torch.tensor(y_val))

    model = MLP(in_features=X[0].shape[-1], out_features=1, hidden=[128], activation=torch.nn.ReLU())
    weight = np.sum(y[mask_train] == 0) / np.sum(y[mask_train] == 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(device))

    val_preds, val_auc, model = trainer(
        model=model,
        criterion=criterion,
        X=X,
        X_ids=X_ids,
        train_set=train_set,
        val_set=val_set,
        params=params,
        device=device,
        padded=False,
        metric_patient_wise=False,
    )
    return val_preds, val_auc, model
