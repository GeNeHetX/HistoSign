import pandas as pd
from pathlib import Path
import numpy as np

from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device
from deepmil import DeepMIL, MLP
from torch.nn import ReLU
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--export_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\inference_panc"),
        help="Path to the directory where the results will be saved",
        required=False,
    )
    parser.add_argument(
        "--model_sign_path",
        type=Path,
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\best_model_path.npy"),
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\classic_basal_model_path.npy"),
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\hwang_model_path.npy"),
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\all_model_path.npy"),
        help="Path the file containing a dictionary whose keys are the class names and the values are the paths to the models",
    )
    parser.add_argument(
        "--model_tum_path",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\trainings\tumors\2024-03-25_14-43-57\model.pth"),
        help="Path to the tumor model",
    )
    parser.add_argument(
        "--device",
        type=device,
        default="cuda:0",
        help="Device to use for the predictions",
        choices=["cuda:0", "cpu"],
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_ctranspath"),
        help="Path to the directory containing the features",
    )
    parser.add_argument(
        "--summary_data",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv"),
        help="Path to the summary data",
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\non_white_coordinates_panc_224"),
        help="Path to the coordinates",
    )

    return parser.parse_args()


def main(args):

    summ_df = pd.read_csv(args.summary_data)

    pbar = tqdm(summ_df.iterrows(), total=len(summ_df), desc="Processing WSIs")
    for i, row in pbar:
        pbar.set_postfix_str(f"Processing {row['sample_ID']}")
        x = np.load(args.features_dir / row["sample_ID"] / "features.npy")
        coord = x[:, :3].astype(int)
        x = x[:, 3:]
        x = torch.from_numpy(x).unsqueeze(0).float()

        df_wsi = pd.DataFrame()
        df_tiles = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2]})
        model_paths_dict = np.load(args.model_sign_path, allow_pickle=True).item()
        for name, path in model_paths_dict.items():
            _df_wsi, _df_tiles = sign_pred(x, coord, name, path, "cuda:0")
            df_wsi = pd.concat([df_wsi, _df_wsi], axis=1)
            df_tiles = pd.merge(df_tiles, _df_tiles, on=["z", "x", "y"], how="outer")

        _df_tum = tum_pred(x, coord, args.model_tum_path, args.device)
        df_tiles = pd.merge(df_tiles, _df_tum, on=["z", "x", "y"], how="outer")

        export_folder = args.export_dir / row["sample_ID"]
        export_folder.mkdir(parents=True, exist_ok=True)
        df_wsi.to_csv(export_folder / "wsi_pred.csv", index=False)
        df_tiles.to_csv(export_folder / "tiles.csv", index=False)


def sign_pred(x, coord, sign_name, model_sign_path, device="cuda:0"):

    model_sign = DeepMIL(
        in_features=x.shape[2],
        out_features=1,
        d_model_attention=128,
        mlp_hidden=[128, 64],
        mlp_activation=ReLU(),
        tiles_mlp_hidden=[128],
    )
    model_sign.load_state_dict(torch.load(model_sign_path))
    model_sign = model_sign.to(device)

    with torch.inference_mode():
        # if this part doesnt fit in memory, use a loop or wrap in a dataloader
        tiles_emb = model_sign.tiles_emb(x.to(device))

        # Compute WSI-level signature predictions
        scaled_tiles_emb, _ = model_sign.attention_layer(tiles_emb, None)
        logits = model_sign.mlp(scaled_tiles_emb)
        logits = logits.cpu().numpy()

        # Compute tile-level signature predictions
        tiles_emb_tile = tiles_emb.reshape(tiles_emb.shape[1], 1, -1)
        scaled_tiles_emb_tile, _ = model_sign.attention_layer(tiles_emb_tile, None)
        logits_tile = model_sign.mlp(scaled_tiles_emb_tile)
        logits_tile = logits_tile.cpu().numpy()

    df_wsi = pd.DataFrame({sign_name: logits[:, 0]})
    df_tiles = pd.DataFrame(
        {"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], sign_name: logits_tile[:, 0]}
    )

    return df_wsi, df_tiles


def tum_pred(x, coord, model_tum_path, device="cuda:0"):
    model_tum = MLP(
        in_features=x.shape[2],
        out_features=1,
        hidden=[128],
        activation=ReLU(),
    )
    model_tum.load_state_dict(torch.load(model_tum_path, map_location="cpu"))
    model_tum.sigmoid = torch.nn.Sigmoid()
    model_tum = model_tum.to(device)

    with torch.inference_mode():
        tum_pred = model_tum(x.to(device))
        tum_pred = tum_pred.cpu().numpy().squeeze()
        _df_tum = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], "tum_pred": tum_pred})
    return _df_tum


if __name__ == "__main__":
    args = parse_arg()
    main(args)
