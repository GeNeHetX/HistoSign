from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold


class SignaturetDataset:
    def __init__(
        self,
        PATH_SUMMARY_DATA: Path,
        PATH_FEATURES_DIR: Path,
        PATH_COL_SIGNS: Path = None,
        PATH_TUM_ANNOT: Path = None,
        col_name: str = None,
    ) -> None:
        self.df = pd.read_csv(PATH_SUMMARY_DATA)
        try:
            self.df = self.df.dropna(
                subset=["histological_aspect", "sous_type_visuel", "stroma_visual_aspect"]
            )
        except KeyError:
            pass
        assert self.df.isna().any().any() == False, "There are NaN values in the dataset"
        # Uncomment the next two line if on Windows
        # self.df.path_svs = self.df.path_svs.apply(lambda x: "D:/" / Path("/".join(x.split("/")[3:])))
        # self.df.path_xml = self.df.path_xml.apply(lambda x: "D:/" / Path("/".join(x.split("/")[3:])))
        self.df.index = self.df.patient_ID
        self.df.sort_index(inplace=True)

        self.PATH_FEATURES_DIR = PATH_FEATURES_DIR
        self.PATH_COL_SIGNS = PATH_COL_SIGNS
        self.PATH_TUM_ANNOT = PATH_TUM_ANNOT
        self.col_name = col_name

        assert self.PATH_FEATURES_DIR.exists(), "The path to the features directory does not exist"

        assert (
            self.col_name is None or self.PATH_COL_SIGNS is None
        ), "You must provide either a col_name or a path to the file containing the column names of the molecular signatures"
        if self.col_name is not None:
            assert self.col_name in self.df.columns, "The column name provided is not in the dataframe"

    def load_sign(self, return_val: str = "short") -> pd.DataFrame:
        """Loads molecular signatures and returns a pandas DataFrame
        where each column correspond to the value of each of the signatures.
        Index must be patient id.

        Parameters
        ----------
        return_val : str, default="short"
            - "short": returns only the Classic and Basal components
            - "normal": returns the Classic, StromaActiv, Basal and StromaInactive components
            - "long": returns all the components
        """

        assert return_val in [
            "short",
            "normal",
            "long",
            "custom",
        ], "return_val must be either 'short' or 'normal' or 'long' or 'custom'"

        if return_val == "long":
            assert (
                self.PATH_COL_SIGNS is not None
            ), "You must provide a path to the file containing the column names of the molecular signatures"
            col_to_pred = np.loadtxt(self.PATH_COL_SIGNS, dtype=str, encoding="utf-8")
            return self.df[col_to_pred].copy()

        elif return_val == "normal":
            return self.df[
                [
                    "Classic",
                    "StromaActiv",
                    "Basal",
                    "StromaInactive",
                ]
            ].copy()

        elif return_val == "short":
            return self.df[
                [
                    "Classic",
                    "Basal",
                ]
            ].copy()
        elif return_val == "custom":
            assert self.col_name is not None, "You must provide a column name"
            return pd.DataFrame(self.df[self.col_name].copy())

    def load_tum_annot(self) -> pd.DataFrame:
        """Loads the tumor annotation file and returns a pandas DataFrame. The columns are the sample_ID, z,x,y,tum"""
        assert self.PATH_TUM_ANNOT is not None, "You must provide a path to the tumor annotation file"

        res = {}
        for k, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading tumor annotations"):
            tile_coords = np.load(self.PATH_TUM_ANNOT / row.sample_ID / "tiles_coord.npy")
            temp_df = pd.DataFrame(
                {
                    "z": tile_coords[:, 0],
                    # "k": tile_coords[:, 1],
                    "x": tile_coords[:, 2],
                    "y": tile_coords[:, 3],
                    "annot": tile_coords[:, 4],
                }
            )
            res[row.sample_ID] = temp_df.copy()
        return res

    def load_features(
        self,
        n_tiles: int = 10_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads features for each slide."""
        features_paths = list(self.PATH_FEATURES_DIR.glob("*/features.npy"))

        features = pd.DataFrame(features_paths, columns=["path"])
        features["sample_ID"] = features["path"].apply(lambda x: x.parents[0].name)

        features = pd.merge(
            features,
            self.df,
            on="sample_ID",
            how="right",
        )

        x = np.load(features.iloc[0].path, mmap_mode="r")[:n_tiles].copy()
        n_features = x.shape[-1]

        X = np.zeros((len(features_paths), n_tiles, n_features), dtype=np.float32)
        X_slidenames, X_ids = [], []

        for i, row in tqdm(features.iterrows(), total=len(features), desc="Loading features"):
            # Load features
            sample_ID = row.sample_ID
            patient_id = row.patient_ID
            # x = np.load(row.path, mmap_mode="r")[:n_tiles].copy()

            x_temp = np.load(row.path, mmap_mode="r")
            # random sampling of the tiles, if possible
            if x_temp.shape[0] > n_tiles:
                idx = np.random.choice(x_temp.shape[0], n_tiles, replace=False)
                x = x_temp[idx]
            else:
                x = x_temp

            if len(x) < n_tiles:
                print(
                    f"Warning: '{sample_ID}' has less than {n_tiles} tiles, only {len(x)} tiles will be used."
                )

            if len(x) == 0:
                raise ValueError("Warning: '{0}' has no tiles, skipping this slide.".format(sample_ID))

            # Fill the global features, slidenames and patient ids matrices
            X[i, : len(x), :] = x
            X_slidenames.append(sample_ID)
            X_ids.append(patient_id)

            # if i > 5:
            #     break

        # return feature, coordinates, slidenames, patient_ids
        return X[..., 3:], X[..., :3], np.array(X_slidenames), np.array(X_ids)

    def get_ids_cv_splits(
        self,
        labels: pd.Series,
        use_cross_val: bool = True,
        use_multicentric: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute cross validation split for the dataset. Uses GroupKFold to ensure that the same patient is not present in both the train and validation set."""

        groups = labels.index.values
        cv_train_ids, cv_val_ids = {}, {}
        if not use_cross_val:
            cv_train_ids["split_0"] = labels.index.values
            cv_val_ids["split_0"] = labels.index.values
            return cv_train_ids, cv_val_ids

        if use_multicentric:
        # basically get training as DISC and validation as BJN_U
            df_disc = self.df[self.df.cohort == "DISC"]
            df_bjn_u = self.df[self.df.cohort == "BJN_U"]
            train_idx = labels.index.isin(df_disc.patient_ID)
            val_idx = labels.index.isin(df_bjn_u.patient_ID)
            cv_train_ids["split_0"] = labels.index[train_idx].values.squeeze()
            cv_val_ids["split_0"] = labels.index[val_idx].values.squeeze()
            return cv_train_ids, cv_val_ids

        gkf = GroupKFold(n_splits=5)
        for i, (train_idx, val_idx) in enumerate(gkf.split(labels, labels, groups)):
            cv_train_ids[f"split_{i}"] = labels.iloc[train_idx].index.values.squeeze()
            cv_val_ids[f"split_{i}"] = labels.iloc[val_idx].index.values.squeeze()

        return cv_train_ids, cv_val_ids
