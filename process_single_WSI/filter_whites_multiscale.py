import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    print("Windows")
    with os.add_dll_directory(OPENSLIDE_PATH):
        print("Added", OPENSLIDE_PATH, "to the DLL directories")
        import openslide
else:
    import openslide

from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage.morphology import disk, binary_closing
from scipy.ndimage import binary_fill_holes
from xml.etree import ElementTree as ET
from argparse import ArgumentParser

from multiprocessing import Pool


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=Path,
        default=r"D:\PACPaint_homemade\datasets\summary_data_full.csv",
        help="Path to the dataframe containing the paths to the slides",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=224,
        help="Size of the tiles to extract",
    )
    parser.add_argument(
        "--log_file_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset\log_filter_whites_"
        + datetime.now().strftime("%Y%m%d")
        + ".txt",
        help="Path to the log file containing the paths to the slides already processed",
    )
    parser.add_argument(
        "--folder_path",
        type=Path,
        default=r"C:\Users\inserm\Documents\histo_sign\dataset\non_white_tiles_coord",
        help="Path to the folder where to save the tiles coordinates",
    )
    return parser.parse_args()


class TilesExtractorDataset(Dataset):
    def __init__(
        self,
        slide: openslide.OpenSlide,
        tile_size: int = 224,
    ) -> None:
        self.slide = slide
        self.dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
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
        else:
            raise ValueError(f"Objective power {self.zoom_level}x not supported")
        self.z = self.level
        self.h, self.w = self.dz.level_dimensions[self.level]
        self.h_tile, self.w_tile = self.dz.level_tiles[self.level]
        # Get rid of the last row and column because can't fit a full tile usually
        self.h_tile -= 1
        self.w_tile -= 1

    def idx_to_ij(self, item: int):
        return np.unravel_index(item, (self.h_tile, self.w_tile))

    def __len__(self) -> int:
        return self.h_tile * self.w_tile


def filter_whites(path_svs, tile_size=224, log_file_path=None, folder_path=None, thumb_size=(1000, 1000)):
    """
    Extract the coordinates of the tiles that are not white in the thumbnail of the WSI.
    We use
    """

    slide = openslide.OpenSlide(path_svs)
    slide_dt = TilesExtractorDataset(slide, tile_size=tile_size)

    # First segment the slide
    img_pil = slide.get_thumbnail(thumb_size)
    img = np.array(img_pil)
    tresh = 220
    mask = img.mean(axis=2) < tresh
    closed = binary_closing(mask, disk(3))
    filled = binary_fill_holes(closed, structure=np.ones((15, 15)))
    # final_mask = binary_closing(closed, disk(3))
    final_mask = binary_closing(filled, disk(3))

    # Get various dimensions
    h_thumb, w_thumb = img.shape[:2]
    # retrieve the dimensions of the wsi at the chosen level
    w_slide, h_slide = slide_dt.dz.level_dimensions[slide_dt.level]
    z = slide_dt.z

    w_ratio = w_thumb / w_slide
    h_ratio = h_thumb / h_slide

    all_tiles_coord = [
        [z, k, i, j] for k, i, j in zip(range(len(slide_dt)), *slide_dt.idx_to_ij(range(len(slide_dt))))
    ]
    all_tiles_coord = np.array(all_tiles_coord)
    # Get the coordinates of the tiles in the thumbnail
    coord_thumb = np.zeros((all_tiles_coord.shape[0], 4, 2), dtype=np.int32)
    for k in tqdm(range(all_tiles_coord.shape[0])):
        # Convert tile adress to pixel coordinates in the full resolution image
        i, j = all_tiles_coord[k, 2] * tile_size, all_tiles_coord[k, 3] * tile_size
        corners = np.array(
            [[i, j], [i + tile_size, j], [i, j + tile_size], [i + tile_size, j + tile_size]]
        ).astype(np.float32)
        # Map the coordinates of the pixels i,j to the coordinates on the thumbnail
        corners[:, 0] *= h_ratio
        corners[:, 1] *= w_ratio
        coord_thumb[k] = np.round(corners).astype(np.int32)

    # Keep only the tiles that are inside the thumbnail
    valid_idx = np.all((coord_thumb[:, :, 0] < w_thumb) & (coord_thumb[:, :, 1] < h_thumb), axis=1)
    coord_thumb = coord_thumb[valid_idx]
    all_tiles_coord = all_tiles_coord[valid_idx]

    # We keep the tile if at least one of its corner is inside the mask
    valid_tiles_idx = final_mask[coord_thumb[:, :, 1], coord_thumb[:, :, 0]].sum(axis=1) > 1
    tiles_coord = all_tiles_coord[valid_tiles_idx]
    coord_thumb = coord_thumb[valid_tiles_idx]

    if folder_path is not None:
        slide_name = Path(path_svs).stem
        export_path = folder_path / f"{slide_name}"
        export_path.mkdir(parents=True, exist_ok=True)
        np.save(export_path / "coord_thumb.npy", coord_thumb)
        np.save(export_path / "final_mask.npy", final_mask)
        np.save(export_path / "img.npy", img)
        np.save(export_path / "tiles_coord.npy", tiles_coord)

    if log_file_path is not None:
        # log the path as completed
        with open(log_file_path, "a") as f:
            f.write(f"{path_svs}\n")
            f.close()
        print(f"Finished process for slide {Path(path_svs).stem} found {len(tiles_coord)} tiles")

    return tiles_coord, coord_thumb, final_mask, img


if __name__ == "__main__":
    start_time = datetime.now()
    args = parse_args()
    print(args)
    df = pd.read_csv(Path(args.df_path))
    if args.log_file_path.exists():
        log = np.loadtxt(args.log_file_path, dtype=str)
        log = np.array([Path(p) for p in log])
    else:
        log = np.array([], dtype=str)

    df = df[~df.path_svs.isin(log)]
    list_of_path_svs = df.path_svs.values

    def custom_iter():
        for path_svs in list_of_path_svs:
            yield path_svs, args.tile_size, args.log_file_path, args.folder_path

    it = custom_iter()
    with Pool(6) as p:
        p.starmap(filter_whites, it)

    print(f"Finished in {datetime.now() - start_time}")
