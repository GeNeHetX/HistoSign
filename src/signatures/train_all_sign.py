from argparse import ArgumentParser
from pathlib import Path
from train import main as train_signatures
import numpy as np
from datetime import datetime


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
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\selected_col_names.txt"),
        # default=None,
        help="Path to the file containing the names of the columns to predict",
    )
    parser.add_argument(
        "--export_folder",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\trainings\single_signatures"),
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


def main(args):
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    col_names = np.loadtxt(args.col_signs, dtype=str, encoding="utf-8")
    PARAMS = {
        "batch_size": args.batch_size,
        "n_ep": args.n_ep,
        "lr": args.lr,
        "n_tiles": args.n_tiles,
        "return_sign": "custom",
        "use_cross_val": args.use_cross_val,
        "n_workers": args.n_workers,
        "wd": args.wd,
    }
    args.export_folder = args.export_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for signature in col_names:
        print(f"Training signature: {signature}...")
        args.col_name = signature
        export_path_sign = args.export_folder / signature
        export_path_sign.mkdir(exist_ok=True, parents=True)
        train_signatures(PARAMS, args.summary_data, args.features_dir, None, signature, export_path_sign)
        print(f"Training signature: {signature}... Done \n")
        print("#" * 80)

    print(f"End time: {datetime.now()}")
    print(f"Training all signatures took {datetime.now() - start_time}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
