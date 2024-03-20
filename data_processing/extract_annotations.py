import os

OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import numpy as np
import openslide
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime


from polygons import Polygon_Opti

# multiprocessing
from multiprocessing import Pool

from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import *


def extract_polygons(layer_id: float, path_xml: Path) -> Tuple[list, float]:
    tree = ET.parse(path_xml)
    root = tree.getroot()
    annotations = root.findall("Annotation")
    res = float(root.attrib["MicronsPerPixel"])
    for layer in annotations:
        # only keep the layer corresponding to the RNASeq
        if int(layer.attrib["Id"]) != layer_id:
            continue
        for region in layer.findall("Regions"):
            list_polygon = []
            for r in region.findall("Region"):
                # sometime multiple polygons are defined for the same layer
                vertices = r.find("Vertices")
                x, y = [], []
                for v in vertices.findall("Vertex"):
                    x.append(float(v.attrib["X"]) * res)
                    y.append(float(v.attrib["Y"]) * res)
                polygon = np.array([x, y]).T
                list_polygon.append(polygon)
            # if len(list_polygon) > 1:
            #   print("Found more than one polygon for sample", row.sample_ID, "in scan", row.ID_scan, "at index", idx)
    return list_polygon, res


def process_sample(
    layer_id: int, path_xml: Path, path_svs: Path, tiles_coord: np.ndarray, tile_size: int = 224
) -> np.ndarray:

    slide = openslide.OpenSlide(str(path_svs))
    zoom_level = int(slide.properties["openslide.objective-power"])
    downsampling_factor = 1 if zoom_level == 20 else 2

    pol_list, resolution = extract_polygons(layer_id, path_xml)
    tiles_coord_micro = tiles_coord.copy().astype(float)
    tiles_coord_micro[:, 2] *= tile_size * resolution * downsampling_factor
    tiles_coord_micro[:, 3] *= tile_size * resolution * downsampling_factor

    pts = tiles_coord_micro[:, 2:4]
    corners = [
        pts,
        pts + [tile_size * resolution, 0],
        pts + [tile_size * resolution, tile_size * resolution],
        pts + [0, tile_size * resolution],
    ]
    corners = np.array(corners)
    valid_tile = np.zeros(tiles_coord.shape[0], dtype=bool)
    for pol in pol_list:
        pol = Polygon_Opti(pol)
        res = pol.are_inside(corners.reshape(-1, 2))
        res = res.reshape(corners.shape[:2]).sum(axis=0) >= 2
        valid_tile = valid_tile | res

    # valid_coord = tiles_coord[valid_tile]

    return valid_tile


def process_row(row_tuple):
    row, path_coord, export_folder, tile_size = row_tuple
    tiles_coord = np.load(path_coord / row.ID_scan / "tiles_coord.npy")

    valid_tile = process_sample(row.Layer_id, row.path_xml, row.path_svs, tiles_coord, tile_size=tile_size)
    valid_coord = tiles_coord[valid_tile]

    export_path = export_folder / row.sample_ID
    export_path.mkdir(exist_ok=True, parents=True)
    np.save(export_path / "tiles_coord.npy", valid_coord)

    return 0


def main():
    start_time = datetime.now()
    tile_size = 224
    path_coord = Path(
        r"C:\Users\inserm\Documents\histo_sign\dataset\non_white_coordinates_mdn_" + str(tile_size)
    )
    export_folder = Path(r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_mdn_" + str(tile_size))

    summ_df = pd.read_csv(r"C:\Users\inserm\Documents\histo_sign\dataset\mdn_summary_uq.csv")
    summ_df.path_svs = summ_df.path_svs.apply(Path)
    summ_df.path_xml = summ_df.path_xml.apply(Path)
    summ_df["Layer_id"] = summ_df.sample_ID.apply(lambda x: x.split("_L")[1].split("_")[0]).astype(int)
    summ_df.sort_values("ID_scan", inplace=True)
    summ_df.reset_index(drop=True, inplace=True)

    def custom_iter():
        for _, row in summ_df.iterrows():
            yield (row, path_coord, export_folder, tile_size)

    with Pool(processes=10) as p:
        max_ = len(summ_df)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(process_row, custom_iter()):
                pbar.update()
    print("Finished in", datetime.now() - start_time)


if __name__ == "__main__":
    main()
