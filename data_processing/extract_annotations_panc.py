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
from polygons import Polygon_Opti
from pathlib import Path
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser

rtol = 3e-02
atol = 40


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=Path,
        help="Path to the csv file containing the data",
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv"),
    )
    parser.add_argument(
        "--path_coord",
        type=Path,
        help="Path to the folder containing the coordinates",
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\non_white_coordinates_panc_224"),
    )
    parser.add_argument(
        "--export_folder",
        type=Path,
        help="Path to the folder where to save the results",
        default=Path(r"C:\Users\inserm\Documents\histo_sign\dataset\coordinates_panc_224"),
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        help="Path to the log file",
        default=Path(r"C:\Users\inserm\Documents\histo_sign")
        / f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
    )
    parser.add_argument("--tile_size", type=int, default=224, help="Size of the tiles")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of jobs to use")
    return parser.parse_args()


def is_closed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.allclose(x[0], x[-1], rtol=rtol, atol=atol) and np.allclose(y[0], y[-1], rtol=rtol, atol=atol)


def all_closed(pol_list):
    res = np.array([is_closed(pol[:, 0], pol[:, 1]) for pol in pol_list])
    return res.all(), res


def fuse_pol(dict_non_closed, dict_roa_non_closed, ref_key=0):
    pol_ref = dict_non_closed[ref_key]
    keys = list(dict_non_closed.keys())

    for key in keys:
        if key == ref_key:
            continue
        pol = dict_non_closed[key]
        fuse = False

        if np.allclose(pol_ref[-1], pol[0], rtol=rtol, atol=atol):
            dict_non_closed[ref_key] = np.concatenate([pol_ref, pol])
            fuse = True
        elif np.allclose(pol_ref[-1], pol[-1], rtol=rtol, atol=atol):
            dict_non_closed[ref_key] = np.concatenate([pol_ref, pol[::-1].copy()])
            fuse = True
        elif np.allclose(pol_ref[0], pol[-1], rtol=rtol, atol=atol):
            dict_non_closed[ref_key] = np.concatenate([pol.copy(), pol_ref])
            fuse = True
        elif np.allclose(pol_ref[0], pol[0], rtol=rtol, atol=atol):
            dict_non_closed[ref_key] = np.concatenate([pol[::-1].copy(), pol_ref])
            fuse = True
        # else:
        #     print("No fusion between the two polygons", ref_key, key)

        if fuse:
            # print(f"Fuse {ref_key=} with {key=}; {len(dict_non_closed)=}")
            del dict_non_closed[key]
            del dict_roa_non_closed[key]
            break

    # if not fuse:
    #     print(f"No fusion for {ref_key=}")

    return dict_non_closed, dict_roa_non_closed, fuse


def join_polygons(dict_non_closed, dict_roa_non_closed, pol_list_closed, neg_roa_closed):
    run = 0
    ref_key = 0
    seen_ref_key = []
    while len(dict_non_closed) > 1:
        dict_non_closed, dict_roa_non_closed, fuse = fuse_pol(dict_non_closed, dict_roa_non_closed, ref_key)
        if run > 100:
            break
        run += 1
        # Find next ref_key
        if not fuse:
            seen_ref_key.append(ref_key)
            for k in dict_non_closed.keys():
                if k not in seen_ref_key:
                    ref_key = k
                    break
            if len(seen_ref_key) == len(dict_non_closed):
                # print("No more ref_key")
                break
        # # Plot
        # if run % 10 == 0:
        #     plt.figure(figsize=(5,5))
        #     for pol in dict_non_closed.values():
        #         plt.plot(pol[:,0],pol[:,1])
        #     plt.show()
    # if len(dict_non_closed) != 1:
    #     print(f"Not all polygons have been fused, {len(dict_non_closed)} left")
    # else:
    #     print(f"All polygons have been fused, {len(dict_non_closed)} left")
    final_pol_list = list(dict_non_closed.values()) + pol_list_closed
    final_neg_roa_list = list(dict_roa_non_closed.values()) + neg_roa_closed
    # print("Are all polygons closed ?", all_closed(final_pol_list))
    return final_pol_list, final_neg_roa_list


def extract_polygons(path_xml):
    tree = ET.parse(path_xml)
    root = tree.getroot()
    res = float(root.attrib["MicronsPerPixel"])

    # Retrieve the polygons
    regions = root.find("Annotation").find("Regions").findall("Region")
    pol_list = []
    neg_roa_list = []
    for r in regions:
        x, y = [], []
        for v in r.find("Vertices").findall("Vertex"):
            x.append(float(v.attrib["X"]) * res)
            y.append(float(v.attrib["Y"]) * res)
        pol_list.append(np.array([x, y]).T)
        neg_roa_list.append(int(r.attrib["NegativeROA"]))
    neg_roa_list = np.array(neg_roa_list)

    # Disciminate between closed and non closed polygons
    pol_list_non_closed = []
    pol_list_closed = []
    neg_roa_non_closed = []
    neg_roa_closed = []
    for pol, roa_label in zip(pol_list, neg_roa_list):
        if not is_closed(pol[:, 0], pol[:, 1]):
            pol_list_non_closed.append(pol)
            neg_roa_non_closed.append(roa_label)
        else:
            pol_list_closed.append(pol)
            neg_roa_closed.append(roa_label)

    # Fuse the non closed polygons
    dict_non_closed = {i: pol for i, pol in enumerate(pol_list_non_closed)}
    dict_roa_non_closed = {i: roa for i, roa in enumerate(neg_roa_non_closed)}

    final_pol_list, final_neg_roa_list = join_polygons(
        dict_non_closed, dict_roa_non_closed, pol_list_closed, neg_roa_closed
    )

    return final_pol_list, final_neg_roa_list, res


def clean_polygons(row, path_to_coord, tile_size=224):
    slide = openslide.OpenSlide(str(row.path_svs))
    zoom_level = int(slide.properties["openslide.objective-power"])
    downsampling_factor = 1 if zoom_level == 20 else 2

    final_pol_list, final_neg_roa_list, resolution = extract_polygons(row.path_xml)

    tiles_coord = np.load(path_to_coord / str(row.sample_ID) / "tiles_coord.npy")
    tiles_coord = np.c_[tiles_coord, np.zeros(len(tiles_coord))]
    tiles_coord_micro = tiles_coord.copy().astype(float)
    tiles_coord_micro[:, 2] *= tile_size * resolution * downsampling_factor
    tiles_coord_micro[:, 3] *= tile_size * resolution * downsampling_factor

    return final_pol_list, final_neg_roa_list, tiles_coord_micro, tiles_coord, resolution


def process_sample(row, path_to_coord, tile_size=224):
    pol_list, neg_roa_list, tiles_coord_micro, tiles_coord, resolution = clean_polygons(row, path_to_coord)
    pts = tiles_coord_micro[:, 2:4]
    corners = [
        pts,
        pts + [tile_size * resolution, 0],
        pts + [tile_size * resolution, tile_size * resolution],
        pts + [0, tile_size * resolution],
    ]
    corners = np.array(corners)

    valid_tile = np.zeros(tiles_coord.shape[0], dtype=bool)
    excluded_tile = np.zeros(tiles_coord.shape[0], dtype=bool)

    # batch process the corners to avoid memory issues
    batch_size = 4096
    n = corners.shape[1]
    n_batch = n // batch_size
    if n % batch_size != 0:
        n_batch += 1
    # print(f"Processing {n} corners in {n_batch} batches, {corners.shape=}, {batch_size=}")
    pbar = tqdm(range(n_batch), total=n_batch, desc="Processing", unit="batch")
    for i in pbar:
    # for i in range(n_batch):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        pts_to_process = corners[:, idx]
        for roa, pol in zip(neg_roa_list, pol_list):
            pol = Polygon_Opti(pol)
            res = pol.are_inside(pts_to_process.reshape(-1, 2))
            res = res.reshape(pts_to_process.shape[:2]).sum(axis=0) >= 2
            if roa:
                excluded_tile[idx] |= res
            else:
                valid_tile[idx] |= res

    # without batch version
    # for roa, pol in zip(neg_roa_list, pol_list):
    #     pol = Polygon_Opti(pol)
    #     res = pol.are_inside(corners.reshape(-1, 2))
    #     res = res.reshape(corners.shape[:2]).sum(axis=0) >= 2
    #     if roa:
    #         excluded_tile |= res
    #     else:
    #         valid_tile |= res

    valid_tile = np.logical_and(valid_tile, ~excluded_tile)
    return tiles_coord, valid_tile


def process_row(row_tuple):
    row, path_coord, export_folder, tile_size, log_path = row_tuple
    # print(f"Processing {row.sample_ID}")

    tiles_coord, valid_tile = process_sample(row, path_coord, tile_size=tile_size)
    tiles_coord[valid_tile, 4] = 1

    export_path = export_folder / row.sample_ID
    export_path.mkdir(exist_ok=True, parents=True)
    np.save(export_path / "tiles_coord.npy", tiles_coord)

    with open(log_path, "a") as f:
        f.write(f"{row.sample_ID}\n")

    return 0


def main(args):
    start_time = datetime.now()
    print(f"Start processing at {start_time}")
    print(args)
    print()

    tile_size = args.tile_size
    path_coord = Path(args.path_coord)
    export_folder = Path(args.export_folder)
    export_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.df_path)
    df.path_svs = df.path_svs.apply(Path)
    df.path_xml = df.path_xml.apply(Path)
    n = len(df)

    if args.log_path.exists():
        log_extraction = np.loadtxt(
            args.log_path,
            dtype=str,
            delimiter="$",
        )
    else:
        log_extraction = np.array([], dtype=str)

    df = df[~df.sample_ID.isin(log_extraction)]
    print(f"Already processed {n - len(df)} samples out of {n} samples. To be processed: {len(df)} samples.")

    rows = df.iterrows()
    rows = [(row[1], path_coord, export_folder, tile_size, args.log_path) for row in rows]

    # with Pool(args.n_jobs) as pool:
    #     for _ in tqdm(pool.imap_unordered(process_row, rows), total=len(rows)):
    #         pass

    pbar = tqdm(enumerate(rows), total=len(rows), desc="Processing", unit="sample")
    for k, row in pbar:
        # print(k, row[0].sample_ID)
        pbar.set_description(f"Processing {row[0].sample_ID}")
        process_row(row)

    #     if k >= 10:
    #         break

    print(f"Finished processing at {datetime.now()}")
    print(f"Processing done in {datetime.now() - start_time}")

    return 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
