from tqdm import tqdm
import pandas as pd
import torch
from deepmil import MLP
from dataset import SignaturetDataset
import numpy as np
from pathlib import Path

PARAMS = {
    "batch_size": 4096,
    "n_ep": 20,
    "lr": 1.0e-4,
    # "n_tiles": 8_000,
    "n_tiles": 20,
    "n_workers": 0,
    "wd": 0,
    "device": "cuda:0",
    "display": True,
    "model_path": "path/to/model.pth",
    "export_path": "path/to/export",
}


PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_ctranspath")
# PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_cnn")
PATH_SUMMARY_DATA = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv")
PATH_TUM_ANNOT = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_panc_224")


def main():
    dataset = SignaturetDataset(
        PATH_SUMMARY_DATA=PATH_SUMMARY_DATA,
        PATH_FEATURES_DIR=PATH_FEATURES_DIR,
        PATH_TUM_ANNOT=PATH_TUM_ANNOT,
    )

    print("Loading data...")
    X_train, y_train, ids_train, X_val, y_val, ids_val = dataset.get_data_tumors(n_tiles=PARAMS["n_tiles"])

    model = MLP(in_features=X_train[0].shape[-1], out_features=1, hidden=[128], activation=torch.nn.ReLU())
    model = model.to(PARAMS["device"])
    model.load_state_dict(torch.load(PARAMS["model_path"]))
    model.sigmoid = torch.nn.Sigmoid()
    model = model.to(PARAMS["device"])
    model.eval()

    unique_samples_train = np.unique(ids_train)
    unique_samples_val = np.unique(ids_val)

    def process_patient(patient, X, y):
        idx_patient = np.where(ids_train == patient)[0]
        X = X_train[idx_patient].unsqueeze(0)
        X = torch.from_numpy(X).to(PARAMS["device"])
        y = y_train[idx_patient]
        preds = model.forward(X).cpu().numpy().squeeze()
        preds = pd.DataFrame({"pred": preds, "y": y})
        preds.to_csv(f"{PARAMS['export_path']}/{patient}/tum_preds.csv", index=False)

    for patient in tqdm(unique_samples_train, desc="Processing train"):
        process_patient(patient, X_train, y_train)
    for patient in tqdm(unique_samples_val, desc="Processing val"):
        process_patient(patient, X_val, y_val)


if __name__ == "__main__":
    main()
