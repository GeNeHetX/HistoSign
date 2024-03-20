import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from time import time
from datetime import timedelta
import os, sys

sys.path.insert(0, os.path.abspath(r"C:\Users\inserm\Documents\histo_sign\src"))
from dataset import SignaturetDataset
from core import train

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--summary_data",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\mdn_summary_vst.csv"),
        help="Path to the summary dataframe",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_mdn_224_ctranspath"),
        help="Path to the directory containing the extracted features of the WSI",
    )
    parser.add_argument(
        "--col_signs",
        type=Path,
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\selected_col_names.txt"),
        default=None,
        help="Path to the file containing the names of the columns to predict",
    )
    parser.add_argument(
        "--return_sign",
        type=str,
        default="long",
        choices=["long", "short", "normal", "custom"],
        help="Type of signature to return. \
        If long, predicts the columns given by the col_signs argument. \
        If short, predicts Classic and Basal. \
        If normal, predicts Classic, StromaActiv, Basal, StromaInactive. \
        If custom, predicts the column given by the col_name argument.",
    )
    parser.add_argument(
        "--col_name",
        type=str,
        # default="Classic",
        default=None,
        help="Name of the column to predict",
    )
    parser.add_argument(
        "--export_path",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\trainings\signatures"),
        help="Path to the directory where the results will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the feature extraction",
    )
    parser.add_argument(
        "--n_ep",
        type=int,
        default=100,
        help="Number of epochs for the training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5.0e-4,
        help="Learning rate for the training",
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        default=64,
        help="Number of tiles to use",
    )
    parser.add_argument(
        "--use_cross_val",
        action="store_true",
        default=True,
        help="Whether to use cross-validation",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        help="Weight decay for the optimizer",
    )

    return parser.parse_args()


# export_path = Path(r"C:\Users\inserm\Documents\histo_sign\trainings")
# export_path = export_path / "signatures" / str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# export_path.mkdir(parents=True, exist_ok=True)
# print("Saving results at \n", export_path, "\n")

# PARAMS = {
#     "batch_size": 32,
#     "n_ep": 100,
#     "lr": 5.0e-4,
#     "n_tiles": 64,
#     "return_sign": "long",
#     "use_cross_val": True,
#     "n_workers": 0,
#     "wd": 5e-4,
# }


# # PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_mdn_512_vit")
# PATH_FEATURES_DIR = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_mdn_224_ctranspath")
# PATH_SUMMARY_DATA = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\mdn_summary_vst.csv")
# # PATH_COL_SIGNS = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\col_names.txt")
# PATH_COL_SIGNS = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\new_col_names.txt")
# # PATH_COL_SIGNS = None


def save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, PATH_COL_SIGNS, col_name, export_path):
    with open(export_path / "params.txt", "w", encoding="utf-8") as f:
        f.write("PARAMS = " + str(PARAMS) + "\n")
        f.write("PATH_SUMMARY_DATA = " + str(PATH_SUMMARY_DATA) + "\n")
        f.write("PATH_FEATURES_DIR = " + str(PATH_FEATURES_DIR) + "\n")
        f.write("PATH_COL_SIGNS = " + str(PATH_COL_SIGNS) + "\n")
        f.write("col_name = " + str(col_name) + "\n")
        f.write("export_path = " + str(export_path) + "\n")


def main(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, PATH_COL_SIGNS, col_name, export_path):
    start_time = time()

    save_params(PARAMS, PATH_SUMMARY_DATA, PATH_FEATURES_DIR, PATH_COL_SIGNS, col_name, export_path)

    # Retrieve data
    dataset = SignaturetDataset(
        PATH_SUMMARY_DATA=PATH_SUMMARY_DATA,
        PATH_FEATURES_DIR=PATH_FEATURES_DIR,
        PATH_COL_SIGNS=PATH_COL_SIGNS,
        col_name=col_name,
    )
    X, _, _, X_ids = dataset.load_features(n_tiles=PARAMS["n_tiles"])
    X = torch.from_numpy(X)
    y = dataset.load_sign(return_val=PARAMS["return_sign"])

    # Split data
    common_ids = set(y.index).intersection(X_ids)
    train_ids_cv, val_ids_cv = dataset.get_ids_cv_splits(
        labels=y.loc[list(common_ids)], use_cross_val=PARAMS["use_cross_val"]
    )

    # Train
    print("\n Starting training... \n")
    val_corrs = {comp: [] for comp in y.columns}
    for i, split in enumerate(train_ids_cv):
        val_preds, val_corrs_, model = train(
            X=X,
            X_ids=X_ids,
            y=y,
            train_ids=train_ids_cv[split],
            val_ids=val_ids_cv[split],
            params=PARAMS,
            device="cuda:0",
        )
        for comp in val_corrs_:
            print(f"Split {i}, {comp} component: corr={val_corrs_[comp]:.3f}")
            val_corrs[comp].append(val_corrs_[comp])

        # break

        # Save models and its metrics
        export_folder = export_path / f"split_{i}"
        export_folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), export_folder / "model.pth")
        np.save(export_folder / "val_corrs.npy", val_corrs_)

    for comp in val_corrs:
        print(f"{comp}: Mean corr={np.mean(val_corrs[comp]):.3f}")

    print(f"Done in {timedelta(seconds=time()-start_time)}")


if __name__ == "__main__":
    args = parse_args()
    PARAMS = {
        "batch_size": args.batch_size,
        "n_ep": args.n_ep,
        "lr": args.lr,
        "n_tiles": args.n_tiles,
        "return_sign": args.return_sign,
        "use_cross_val": args.use_cross_val,
        "n_workers": args.n_workers,
        "wd": args.wd,
    }
    args.export_path = args.export_path / str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.export_path.mkdir(parents=True, exist_ok=True)

    main(PARAMS, args.summary_data, args.features_dir, args.col_signs, args.col_name, args.export_path)
    # main(
