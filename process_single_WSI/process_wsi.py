import os

# Demerdez-vous pour installer openslide sur votre machine
OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from display_wsi import display_wsi

from pathlib import Path
from torch import device

from argparse import ArgumentParser

from torch import device
from filter_whites_multiscale import filter_whites
from feature_extraction import extract_features
from deepmil import DeepMIL, MLP
from torch.nn import ReLU
import torch
import pandas as pd
import numpy as np


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--temp_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\temp_folder"),
        help="Path to the temporary directory where the features will be saved",
        required=False,
    )
    parser.add_argument(
        "--wsi",
        type=Path,
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\MDN\12AG00001-14_MDNF01_HES.svs"),
        default=Path(r"D:\PACPaint_homemade\datasets\BJN_U\364842-06.svs"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--model_sign_path",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\best_model_path.npy"),
        help="Path the file containing a dictionary whose keys are the class names and the values are the paths to the models",
    )
    parser.add_argument(
        "--model_tum_path",
        type=Path,
        default=Path(
            r"D:\PACPaint_homemade\pacpaint_tb\..\trainings\neo_cell_type\2024-02-19_09-44-31\split_21\model.pth"
        ),
        help="Path to the tumor model",
    )
    parser.add_argument(
        "--device",
        type=device,
        default="cuda:0",
        help="Device to use for the predictions",
        choices=["cuda:0", "cpu"],
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for the feature extraction")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the feature extraction. Set to 0 if using windows.",
    )
    parser.add_argument("--display", action="store_true", help="Display the WSI and the tiles", default=True)

    parsed_args = parser.parse_args()
    return parsed_args


def main(args):
    slidename = args.wsi.stem
    print("Filtering white tiles...")

    tiles_coord, coord_thumb, final_mask, img = filter_whites(
        args.wsi, tile_size=224, folder_path=args.temp_dir
    )

    print("Extracting features...")
    if not (args.temp_dir / slidename / "features.npy").exists():
        features = extract_features(
            args.wsi,
            model_key="ctrans",
            device=args.device,
            tiles_coords=tiles_coord,
            num_workers=args.num_workers,
            save_folder=args.temp_dir,
            batch_size=args.batch_size,
        )
    if not (args.temp_dir / slidename / "features_tum.npy").exists():
        features_tum = extract_features(
            args.wsi,
            model_key="cnn",
            device=args.device,
            tiles_coords=tiles_coord,
            num_workers=args.num_workers,
            save_folder=args.temp_dir,
            batch_size=args.batch_size,
            filename="features_tum.npy",
        )
    features = np.load(args.temp_dir / slidename / "features.npy")
    features_tum = np.load(args.temp_dir / slidename / "features_tum.npy")

    print("Predicting signatures...")
    x, coord = features[:, 3:], features[:, 1:3]
    x = torch.from_numpy(x).unsqueeze(0).float()

    model_paths_dict = np.load(args.model_sign_path, allow_pickle=True).item()

    df_wsi = pd.DataFrame()
    df_tiles = pd.DataFrame({"x": coord[:, 0], "y": coord[:, 1]})
    for name, path in model_paths_dict.items():
        _df_wsi, _df_tiles = sign_pred(x, coord, name, path, args.device)
        df_wsi = pd.concat([df_wsi, _df_wsi], axis=1)
        df_tiles = pd.merge(df_tiles, _df_tiles, on=["x", "y"], how="outer")

    print("Predicting tumor...")
    x, coord = features_tum[:, 3:], features_tum[:, 1:3]
    x = torch.from_numpy(x).unsqueeze(0).float()
    _df_tum = tum_pred(x, coord, args.model_tum_path, args.device)
    df_tiles = pd.merge(df_tiles, _df_tum, on=["x", "y"], how="outer")

    df_tiles.to_csv(args.temp_dir / slidename / "tiles_preds.csv", index=False)
    df_wsi.to_csv(args.temp_dir / slidename / "wsi_preds.csv", index=False)

    print("Done")

    if args.display:
        display_wsi(
            img,
            final_mask,
            coord_thumb,
            df_tiles,
            name=slidename,
        )


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
        _df_tum = pd.DataFrame({"x": coord[:, 0], "y": coord[:, 1], "tum_pred": tum_pred})
    return _df_tum


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
    df_tiles = pd.DataFrame({"x": coord[:, 0], "y": coord[:, 1], sign_name: logits_tile[:, 0]})

    return df_wsi, df_tiles


if __name__ == "__main__":
    args = parse_arg()
    main(args)
