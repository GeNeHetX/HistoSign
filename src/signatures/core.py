from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from deepmil import DeepMIL
from trainer import trainer


def train(
    X: torch.Tensor,
    X_ids: np.ndarray,
    y: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    params: dict,
    device="cuda:0",
) -> Tuple[pd.DataFrame, pd.DataFrame, float, torch.nn.Module]:
    mask_val = np.array([i in val_ids for i in X_ids])
    mask_train = np.array([i in train_ids for i in X_ids])

    idx_train = np.where(mask_train)[0]
    idx_val = np.where(mask_val)[0]

    X_ids_train = X_ids[mask_train]
    X_ids_val = X_ids[mask_val]
    if params["use_cross_val"]:
        assert len(set(X_ids_train).intersection(X_ids_val)) == 0

    # y_train = y.loc[X_ids_train].values.squeeze()
    # y_val = y.loc[X_ids_val].values.squeeze()

    y_train = y.loc[np.unique(X_ids_train)].values  # .squeeze()
    y_val = y.loc[np.unique(X_ids_val)].values  # .squeeze()

    train_set = TensorDataset(torch.tensor(idx_train), torch.tensor(y_train, dtype=torch.float32))
    val_set = TensorDataset(torch.tensor(idx_val), torch.tensor(y_val, dtype=torch.float32))

    criterion = torch.nn.MSELoss()

    # model = Weldon(
    #     # in_features=2048,
    #     # out_features=4,
    #     in_features=X.shape[2],
    #     out_features=y.shape[1],
    #     n_extreme=100,
    #     tiles_mlp_hidden=[128],
    # )
    model = DeepMIL(
        in_features=X.shape[2],
        out_features=y.shape[1],
        d_model_attention=128,
        mlp_hidden=[128, 64],
        mlp_activation=torch.nn.ReLU(),
        tiles_mlp_hidden=[128],
    )

    val_preds, val_corrs, model = trainer(
        model=model,
        criterion=criterion,
        X=X,
        X_ids=X_ids,
        train_set=train_set,
        val_set=val_set,
        params=params,
        device=device,
        target_names=y.columns.tolist(),
    )

    return val_preds, val_corrs, model
