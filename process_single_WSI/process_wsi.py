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
        # default=Path(r"D:\PACPaint_homemade\datasets\BJN_U\364842-06.svs"),
        # default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\MDN\552138-25_MDNF02_HES.svs"),
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\PRODIGE_24\14AG02095-32_HES.svs"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    parser.add_argument(
        "--ctranspath",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\ctranspath.pth"),
        help="Path to the ctrans model",
    )
    parser.add_argument(
        "--model_sign_path",
        type=Path,
        # default=Path(r"dataset\all_sign.txt"),
        default=Path(r"dataset\best_sign.txt"),
        help="Path to the text file containing the signature names to be predicted",
    )
    parser.add_argument(
        "--model_tum_path",
        type=Path,
        default=Path(r"dataset\model_tum_seg.pth"),
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


def get_models_paths(model_sign_path, folder_path="dataset/models"):
    sign_name = np.loadtxt(model_sign_path, dtype=str, encoding="utf-8")
    model_dict = {sign: Path(folder_path) / sign / "model.pth" for sign in sign_name}
    # verify all models exist
    for sign, path in model_dict.items():
        if not path.exists():
            raise FileNotFoundError(f"Model {sign} not found at {path}")
    return model_dict


def main(args):
    slidename = args.wsi.stem
    print("Filtering white tiles...")

    tiles_coord, coord_thumb, final_mask, img = filter_whites(
        str(args.wsi), tile_size=224, folder_path=args.temp_dir
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
            model_path=args.ctranspath,
        )
    features = np.load(args.temp_dir / slidename / "features.npy")

    x, coord = features[:, 3:], features[:, :3]
    x = torch.from_numpy(x).unsqueeze(0).float()

    print("Predicting signatures...")
    model_paths_dict = get_models_paths(args.model_sign_path)

    df_wsi = pd.DataFrame()
    df_tiles = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2]})
    for name, path in model_paths_dict.items():
        _df_wsi, _df_tiles = sign_pred(x, coord, name, path, args.device)
        df_wsi = pd.concat([df_wsi, _df_wsi], axis=1)
        df_tiles = pd.merge(df_tiles, _df_tiles, on=["z", "x", "y"], how="outer")

    print("Predicting tumor...")
    _df_tum = tum_pred(x, coord, args.model_tum_path, args.device)
    df_tiles = pd.merge(df_tiles, _df_tum, on=["z", "x", "y"], how="outer")

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
        _df_tum = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], "tum_pred": tum_pred})
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
    df_tiles = pd.DataFrame(
        {"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], sign_name: logits_tile[:, 0]}
    )

    return df_wsi, df_tiles


if __name__ == "__main__":
    args = parse_arg()
    main(args)
