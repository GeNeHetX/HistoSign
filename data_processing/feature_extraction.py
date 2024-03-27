from pathlib import Path
import os

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide.deepzoom import DeepZoomGenerator

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.models.resnet import Bottleneck, ResNet
from timm.models.vision_transformer import VisionTransformer

from ctran import ctranspath

from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset\mdn_summary_uq.csv",
        help="Path to the dataframe containing the paths to the slides",
    )
    parser.add_argument(
        "--log_file_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset"
        + "\log_feature_extraction"
        + datetime.now().strftime("%Y%m%d")
        + ".txt",
        help="Path to the log file containing the paths to the slides already processed",
    )
    parser.add_argument(
        "--export_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset\features_mdn_vit",
        help="Path to the folder where to save the features",
    )
    parser.add_argument(
        "--tiles_coord_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_mdn",
        help="Path to the folder where to save the tiles coordinates",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        help="Model to use for the feature extraction",
        choices=["vit", "cnn", "ctrans"],
    )
    parser.add_argument(
        "--path_ctranspath",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\histo_sign\ctranspath.pth"),
        help="Path to the pretrained ctranspath model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for the feature extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Prefetch factor for the dataloader",
    )
    return parser.parse_args()


class TilesDataset(Dataset):
    def __init__(
        self, slide: openslide.OpenSlide, tiles_coords: np.ndarray, key_trans: str = "homemade"
    ) -> None:
        self.slide = slide
        self.tiles_coords = tiles_coords
        if key_trans == "imagenet":
            self.transform = Compose(
                [
                    # ToTensor(),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )  # ImageNet normalization
        elif key_trans == "bt":
            self.transform = Compose(
                [
                    # ToTensor(),
                    Normalize(
                        mean=(0.70322989, 0.53606487, 0.66096631),
                        std=(0.21716536, 0.26081574, 0.20723464),
                    ),
                ]
            )  # Specific normalization for BT -> no idea where these values come from
        elif key_trans == "homemade":
            self.transform = Compose(
                [
                    # ToTensor(),
                    Normalize(
                        mean=(0.8734, 0.7730, 0.7974),
                        std=(0.0846, 0.1234, 0.1127),
                    ),
                ]
            )
            # Statistics calculated from the training set. See the notebook "visualize_tiles.ipynb"
        else:
            raise ValueError(f"Key {key_trans} not recognized. Expected 'imagenet' or 'bt' or 'homemade'.")

        self.dz = DeepZoomGenerator(slide, tile_size=224, overlap=0)

        file_extension = Path(self.slide._filename).suffix
        if file_extension == ".svs":
            self.zoom_level = int(self.slide.properties["openslide.objective-power"])
        elif file_extension == ".qptiff":
            r = (
                ET.fromstring(slide.properties["openslide.comment"])
                .find("ScanProfile")
                .find("root")
                .find("ScanResolution")
            )
            self.zoom_level = float(r.find("Magnification").text)
        elif file_extension == ".ndpi":
            self.zoom_level = int(self.slide.properties["openslide.objective-power"])
        else:
            raise ValueError(f"File extension {file_extension} not supported")

        # We want the second highest level so as to have 112 microns tiles / 0.5 microns per pixel
        if self.zoom_level == 20:
            self.level = self.dz.level_count - 1
        elif self.zoom_level == 40:
            self.level = self.dz.level_count - 2
            self.zoom_level = 20
        else:
            raise ValueError(f"Objective power {self.zoom_level}x not supported")

        assert np.all(
            self.tiles_coords[:, 0] == self.level
        ), "The resolution of the tiles is not the same as the resolution of the slide."
        self.z = self.level

    def __getitem__(self, item: int):
        tile_coords = self.tiles_coords[item, 2:4].astype(int)
        # copy the tile to avoid memory leaks
        tile_coords = np.array(tile_coords, copy=True)

        try:
            im = self.dz.get_tile(level=self.level, address=tile_coords)
        except ValueError:
            print(f"ValueError: impossible to open tile {tile_coords} from {self.slide}")
            raise ValueError

        im = ToTensor()(im)

        if im.shape != torch.Size([3, 224, 224]):
            print(f"Image shape is {im.shape} for tile {tile_coords}. Padding...")
            # PAD the image in white to reach 224x224
            im = torch.nn.functional.pad(im, (0, 224 - im.shape[2], 0, 224 - im.shape[1]), value=1)

        im = self.transform(im)

        return im

    def __len__(self) -> int:
        return len(self.tiles_coords)


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=[2, 3])  # globalpool

        return x


def get_pretrained_url_vit(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    if pretrained:
        pretrained_url = get_pretrained_url_vit(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model


def get_pretrained_url_cnn(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50_special(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url_cnn(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model


def get_ctranspath(model_path):
    model = ctranspath()
    model.head = torch.nn.Identity()
    td = torch.load(model_path)
    model.load_state_dict(td["model"], strict=True)
    return model


def extract_features(
    slide: openslide.OpenSlide,
    model: torch.nn.Module,
    tiles_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    key_trans: str = "homemade",
    num_workers: int = 0,  # num_workers=0 is necessary when using windows
    prefetch_factor: int = None,
) -> np.ndarray:
    dataset = TilesDataset(slide=slide, tiles_coords=tiles_coords, key_trans=key_trans)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    features = []
    with torch.inference_mode():
        for images in tqdm(dataloader, total=len(dataloader)):
            features_b = model(images.to(device))
            features_b = features_b.cpu().numpy()
            features.append(features_b)
    features = np.concatenate(features)
    features = np.concatenate([tiles_coords[:, [0, 2, 3]], features], axis=1)
    # features is of shape (n_tiles, 3 + embed) where the first columns is the resolution, the 2nd and 3rd are the coordinates of the tile, and the rest are the features

    return features


if __name__ == "__main__":
    start_time = datetime.now()
    print("Start time:", datetime.strftime(start_time, "%H:%M:%S"))

    args = parse_args()
    print(args)
    df = pd.read_csv(Path(args.df_path))
    n = len(df)
    if args.log_file_path.exists():
        log_extraction = np.loadtxt(
            args.log_file_path,
            dtype=str,
            delimiter="$",
        )
    else:
        log_extraction = np.array([], dtype=str)
    # log_extraction = [Path(path) for path in log_extraction]
    print("Number of slides processed:", len(log_extraction), "out of", len(df))
    df = df.loc[~df.sample_ID.isin(log_extraction)]  # -> mdn
    # df = df.loc[~df.ID_scan.isin(log_extraction)] # -> prodige_24
    print("Number of slides to process:", len(df))

    device = torch.device(args.device)
    if args.model == "vit":
        model = vit_small(pretrained=True, progress=True, key="DINO_p16").to(device)
        key_trans = "homemade"
    elif args.model == "cnn":
        model = resnet50_special(pretrained=True, progress=True, key="BT").to(device)
        key_trans = "bt"
    elif args.model == "ctrans":
        model = get_ctranspath(args.path_ctranspath).to(device)
        key_trans = "imagenet"
    else:
        raise ValueError(f"Model not recognized: {args.model}")

    model.eval()

    for k, row in df.iterrows():
        print(f"Processing slide {k}/{n} : {row.sample_ID}")
        slide = openslide.OpenSlide(row.path_svs)
        tiles_coord = np.load(args.tiles_coord_path / row.sample_ID / "tiles_coord.npy")
        features = extract_features(
            slide,
            model,
            tiles_coord,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            key_trans=key_trans,
        )

        export_path = args.export_path / row.sample_ID / "features.npy"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(export_path, features)

        with open(args.log_file_path, "a") as f:
            f.write(f"{row.sample_ID}\n")

        # if k > 1:
        #     break

    end_time = datetime.now()
    delta_t = end_time - start_time
    print("End time:", datetime.strftime(end_time, "%H:%M:%S"))
    print("Total time:", str(delta_t))
    print("Finished.")
