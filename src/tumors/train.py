from pathlib import Path
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.abspath(r"C:\Users\inserm\Documents\histo_sign\src"))

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from dataset import SignaturetDataset
from tumors.core import train


export_path = Path(r"C:\Users\inserm\Documents\histo_sign\trainings")
export_path = export_path / "tumors" / str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
export_path.mkdir(parents=True, exist_ok=True)
print("Saving results at \n", export_path, "\n")

PARAMS = {
    "batch_size": 2048,
    "n_ep": 20,
    "lr": 1.0e-4,
    "n_tiles": 8_000,
    "use_cross_val": True,
    "n_workers": 0,
    "wd": 0,
    "use_multicentric": True,
}


PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_ctranspath")
PATH_SUMMARY_DATA = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv")
# PATH_SUMMARY_DATA = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\disc_summary_vst.csv")
PATH_TUM_ANNOT = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_panc_224")


def save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, export_path):
    with open(export_path / "params.txt", "w", encoding="utf-8") as f:
        f.write("PARAMS = " + str(PARAMS) + "\n")
        f.write("PATH_SUMMARY_DATA = " + str(PATH_SUMMARY_DATA) + "\n")
        f.write("PATH_FEATURES_DIR = " + str(PATH_FEATURES_DIR) + "\n")
        f.write("export_path = " + str(export_path) + "\n")

def save_splits(export_path, train_ids_cv, val_ids_cv):
    train_patient = {}
    val_patient = {}
    for split in train_ids_cv:
        train_patient[split] = np.unique(train_ids_cv[split])
        val_patient[split] = np.unique(val_ids_cv[split])
    with open(export_path / "splits.txt", "w", encoding="utf-8") as f:
        f.write("train_patient = " + str(train_patient) + "\n")
        f.write("val_patient = " + str(val_patient) + "\n")


def main():

    start_time = datetime.now()
    print(f"Start time: {start_time}")

    save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, export_path)

    dataset = SignaturetDataset(
        PATH_SUMMARY_DATA=PATH_SUMMARY_DATA,
        PATH_FEATURES_DIR=PATH_FEATURES_DIR,
        PATH_TUM_ANNOT=PATH_TUM_ANNOT,
    )
    X_, _, X_slidenames_, X_ids_ = dataset.load_features(n_tiles=PARAMS["n_tiles"])

    annots = dataset.load_tum_annot()

    X, X_ids, y = [], [], []
    for slidename in tqdm(annots, desc="Filtering features"):
        idx_slide = np.where(X_slidenames_ == slidename)[0][0]

        x = X_[idx_slide]
        x_id = X_ids_[idx_slide]

        # remove padding
        mask_not_padded = x.sum(-1) != 0.0
        x = x[mask_not_padded]

        X.append(x)
        X_ids.extend([x_id] * len(x))
        y.append(annots[slidename].iloc[: len(x)]["annot"].values)

    X = np.concatenate(X, 0)
    X = torch.from_numpy(X)
    X_ids = np.array(X_ids)
    y = pd.Series(np.concatenate(y, 0), index=X_ids)

    common_ids = set(y.index).intersection(X_ids)
    train_ids_cv, val_ids_cv = dataset.get_ids_cv_splits(
        labels=y.loc[list(common_ids)],
        use_cross_val=PARAMS["use_cross_val"],
        use_multicentric=PARAMS["use_multicentric"],
    )
    save_splits(export_path, train_ids_cv, val_ids_cv)
    # del X_, X_slidenames_, X_ids_
    print("\nStarting training :")

    val_aucs = []
    for i, split in enumerate(train_ids_cv):
        val_preds, val_auc, model = train(
            X=X,
            X_ids=X_ids,
            y=y,
            train_ids=train_ids_cv[split],
            val_ids=val_ids_cv[split],
            params=PARAMS,
            device="cuda:0",
        )

        print(f"Split {i}: AUC={val_auc:.3f}")
        val_aucs.append(val_auc)

        # Save models and its metrics
        export_folder = export_path / f"split_{i}"
        export_folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), export_folder / "model.pth")
        np.save(export_folder / "val_auc.npy", val_auc)

    print(f"Mean AUC={np.mean(val_aucs):.3f}")

    print(f"End time: {datetime.now()}. Finished in {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
