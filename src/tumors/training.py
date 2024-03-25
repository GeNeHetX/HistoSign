from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys, os

sys.path.insert(0, os.path.abspath(r"C:\Users\inserm\Documents\histo_sign\src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset import SignaturetDataset
from deepmil import MLP

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

export_path_folder = Path(r"C:\Users\inserm\Documents\histo_sign\trainings")
export_path = export_path_folder / "tumors" / str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
export_path.mkdir(parents=True, exist_ok=True)
print("Saving results at \n", export_path, "\n")

PARAMS = {
    "batch_size": 4096,
    "n_ep": 5,
    "lr": 1.0e-4,
    "n_tiles": 8_000,
    "n_workers": 0,
    "wd": 0,
    "device": "cuda:0",
    "display": True,
}


PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_ctranspath")
# PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_cnn")
PATH_SUMMARY_DATA = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv")
PATH_TUM_ANNOT = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_panc_224")


def save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, export_path):
    with open(export_path / "params.txt", "w", encoding="utf-8") as f:
        f.write("PARAMS = " + str(PARAMS) + "\n")
        f.write("PATH_SUMMARY_DATA = " + str(PATH_SUMMARY_DATA) + "\n")
        f.write("PATH_FEATURES_DIR = " + str(PATH_FEATURES_DIR) + "\n")
        f.write("export_path = " + str(export_path) + "\n")


def save_splits(export_path, train_ids_cv, val_ids_cv):
    train_patient = np.unique(train_ids_cv)
    val_patient = np.unique(val_ids_cv)
    with open(export_path / "splits.txt", "w", encoding="utf-8") as f:
        f.write("train_patient = " + str(train_patient) + "\n")
        f.write("val_patient = " + str(val_patient) + "\n")


def main():

    start_time = datetime.now()
    print(f"Start time: {start_time}")
    save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, export_path)

    # Load data
    dataset = SignaturetDataset(
        PATH_SUMMARY_DATA=PATH_SUMMARY_DATA,
        PATH_FEATURES_DIR=PATH_FEATURES_DIR,
        PATH_TUM_ANNOT=PATH_TUM_ANNOT,
    )

    print("Loading data...")
    # X_train, y_train, ids_train, X_val, y_val, ids_val = dataset.get_data_tumors(n_tiles=PARAMS["n_tiles"])
    # save_splits(export_path, ids_train, ids_val)

    # np.save(export_path_folder / "X_train_panc.npy", X_train)
    # np.save(export_path_folder / "y_train_panc.npy", y_train)
    # np.save(export_path_folder / "X_val_panc.npy", X_val)
    # np.save(export_path_folder / "y_val_panc.npy", y_val)

    X_train = np.load(export_path_folder / "X_train_panc.npy")
    y_train = np.load(export_path_folder / "y_train_panc.npy")
    X_val = np.load(export_path_folder / "X_val_panc.npy")
    y_val = np.load(export_path_folder / "y_val_panc.npy")

    print("Done.")

    # Setup training
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dataloader = DataLoader(
        train_set,
        shuffle=True,
        pin_memory=False,
        batch_size=PARAMS["batch_size"],
        num_workers=PARAMS["n_workers"],
        drop_last=False,
    )
    dataloader_val = DataLoader(
        val_set,
        shuffle=False,
        pin_memory=False,
        batch_size=PARAMS["batch_size"],
        num_workers=PARAMS["n_workers"],
        drop_last=False,
    )
    print(f"Train set: {len(train_set)} tiles, Val set: {len(val_set)} tiles")

    model = MLP(in_features=X_train[0].shape[-1], out_features=1, hidden=[128], activation=torch.nn.ReLU())
    model = model.to(PARAMS["device"])
    weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(PARAMS["device"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["wd"])

    # Training
    val_auc, val_loss, loss_train, auc_train = np.nan, np.nan, np.nan, np.nan
    loss_train_list, val_loss_list = [], []
    auc_train_list, val_auc_list = [], []
    pbar = tqdm(total=len(dataloader))
    for epoch in range(PARAMS["n_ep"]):
        pbar.reset()
        pbar.set_description(
            f"Epoch[{epoch}]: val_loss: {val_loss:.2f}, val_auc: {val_auc:.2f}, loss_train: {loss_train:.2f}"
        )

        model.train()
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(PARAMS["device"]), y_b.to(PARAMS["device"])
            optimizer.zero_grad()
            y_hat = model(x_b.float())
            loss_train = criterion(y_hat.squeeze(), y_b.float())
            loss_train.backward()
            optimizer.step()

            auc_train = roc_auc_score(y_b.cpu(), torch.sigmoid(y_hat).detach().cpu())
            pbar.set_description(
                f"Epoch[{epoch}]: val_loss : {val_loss:.2f}, val_auc: {val_auc:.2f}, train_auc: {auc_train:.2f}, loss_train: {loss_train:.2f}",
                refresh=True,
            )
            pbar.update(1)

        val_auc, val_loss = eval(model, criterion, dataloader_val, device=PARAMS["device"])

        loss_train_list.append(loss_train.cpu().item())
        val_loss_list.append(val_loss)
        auc_train_list.append(auc_train)
        val_auc_list.append(val_auc)
        torch.save(model.state_dict(), export_path / "model.pth")
    pbar.close()

    # Last final evaluation
    print("Training finished. Evaluating final model...")
    val_auc, val_loss = eval(
        model, criterion, dataloader_val, device=PARAMS["device"], with_progess="Validation"
    )
    auc_train, train_loss = eval(
        model, criterion, dataloader, device=PARAMS["device"], with_progess="Training"
    )
    loss_train_list.append(loss_train.cpu().item())
    val_loss_list.append(val_loss)
    auc_train_list.append(auc_train)
    val_auc_list.append(val_auc)
    pbar.set_description(
        f"Epoch[{epoch}]: val_loss : {val_loss:.2f}, val_auc: {val_auc:.2f}, train_auc: {auc_train:.2f}, loss_train: {loss_train:.2f}",
        refresh=True,
    )
    print(f"Final val_auc: {val_auc:.2f}, val_loss: {val_loss:.2f}")
    print(f"Final train_auc: {auc_train:.2f}, train_loss: {train_loss:.2f}")

    # Save model and its metric
    res_dict = {
        "val_auc": val_auc,
        "val_loss": val_loss,
        "train_auc": auc_train,
        "train_loss": train_loss,
    }
    res_df = pd.DataFrame(res_dict, index=[0])
    res_df.to_csv(export_path / "results.csv", index=False)

    torch.save(model.state_dict(), export_path / "model.pth")
    np.save(export_path / "loss_train.npy", np.array(loss_train_list))
    np.save(export_path / "val_loss.npy", np.array(val_loss_list))
    np.save(export_path / "auc_train.npy", np.array(auc_train_list))
    np.save(export_path / "val_auc.npy", np.array(val_auc_list))


    print(f"End time: {datetime.now()}. Finished in {datetime.now() - start_time}")

    if PARAMS["display"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].plot(loss_train_list)
        axes[0, 0].set_title("Train loss")
        axes[0, 1].plot(val_loss_list)
        axes[0, 1].set_title("Val loss")
        axes[1, 0].plot(auc_train_list)
        axes[1, 0].set_title("Train AUC")
        axes[1, 1].plot(val_auc_list)
        axes[1, 1].set_title("Val AUC")
        plt.tight_layout()
        plt.savefig(export_path / "metrics.png")
        plt.show()


def eval(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device="cuda:0",
    with_progess: str = None,
):

    model.eval()
    with torch.no_grad():
        
        if with_progess is not None:
            pbar = tqdm(dataloader, desc=with_progess, total=len(dataloader), unit="batch")
        else:
            pbar = dataloader

        y, y_hat, logits = [], [], []
        for x_b, y_b in pbar:
            logits_b = model(x_b.to(device)).squeeze()
            pred_b = torch.sigmoid(logits_b).squeeze()
            y.append(y_b)
            y_hat.append(pred_b)
            logits.append(logits_b)

        # Loss
        y = torch.cat(y).to(device).float()
        y_hat = torch.cat(y_hat).to(device)
        logits = torch.cat(logits).to(device)
        loss = criterion(logits, y).item()
        auc = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_hat.cpu().numpy())

    return auc, loss


if __name__ == "__main__":
    main()
