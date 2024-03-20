from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from tqdm import tqdm
import pickle
import pandas as pd
import torch
import sys, os

sys.path.insert(0, os.path.abspath(r"C:\Users\inserm\Documents\histo_sign\src"))

from signatures.deepmil import MLP

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--path_features_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\PACPaint_homemade\datasets\features_BT"),
        help="Path to the directory containing the extracted features of the WSI",
    )
    parser.add_argument(
        "--path_summary_data",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\datasets\summary_data_full.csv"),
        help="Path to the summary dataframe",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(r"D:\PACPaint_homemade\trainings\tumors\2024-01-16_11-20-02\split_11\model.pth"),
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path(r"C:\Users\inserm\Documents\PACPaint_homemade\datasets\neo_features_bt_16-01-2024"),
        help="Path to the directory where the predictions will be saved",
    )
    return parser.parse_args()


class InferenceDataset:
    def __init__(self, args) -> None:
        self.args = args
        self.df = pd.read_csv(self.args.path_summary_data)
        self.df.path_svs = self.df.path_svs.apply(Path)
        self.df.path_xml = self.df.path_xml.apply(Path) if "path_xml" in self.df.columns else None
        self.df.index = self.df.sample_ID
        self.df.sort_index(inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # p = list((self.args.path_features_dir / row.sample_ID).glob("*"))[0]
        p = self.args.path_features_dir / row.sample_ID / "features.npy"
        x_full = np.load(p, mmap_mode="r")
        coord = np.array(x_full[:, :3])
        x = np.array(x_full[:, 3:])
        slidename = p.parents[0].name
        return x, coord, slidename


def main():
    args = parse_args()
    print(args)
    args.save_dir.mkdir(exist_ok=True)
    file = open(args.save_dir / "args.txt", "w")
    print(args, file=file)
    preds = {}
    dataset = InferenceDataset(args)
    x, _, _ = dataset[0]
    model = MLP(
        in_features=x.shape[-1],
        out_features=1,
        hidden=[128],
        activation=torch.nn.ReLU(),
    )
    model.load_state_dict(torch.load(args.model_path))
    # Add a sigmoid layer to the model
    model.sigmoid = torch.nn.Sigmoid()
    model.to("cuda")
    model.eval()

    pbar = tqdm(dataset, total=len(dataset))
    for x, coord, slidename in pbar:
        pbar.set_description(f"Processing {slidename}")
        x = torch.from_numpy(x).float().to("cuda")
        with torch.inference_mode():
            preds_ = model.forward(x)
            preds_ = preds_.cpu().numpy().squeeze()
            preds_ = pd.DataFrame({"z": coord[:, 0], "x": coord[:, 1], "y": coord[:, 2], "pred": preds_})
            preds_.to_csv(f"{args.save_dir}/{slidename}.csv", index=False)

            preds[slidename] = preds_

    # pickle.dump(preds, open(f"{args.save_dir}/tumor_preds.pkl", "wb"))


if __name__ == "__main__":
    main()
