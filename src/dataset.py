from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed


class SignaturetDataset:
    def __init__(
        self,
        PATH_SUMMARY_DATA: Path,
        PATH_FEATURES_DIR: Path,
        PATH_COL_SIGNS: Path = None,
        PATH_TUM_ANNOT: Path = None,
        col_name: str = None,
    ) -> None:
        """
        Dataset class for signature prediction and tumor segmentation tasks.

        Parameters
        ----------
        PATH_SUMMARY_DATA : Path
            Path to the dataframe containing the summary data. Must contain the columns `patient_ID` and `sample_ID` as well as the signature columns.
        PATH_FEATURES_DIR : Path
            Path to the directory containing the features of the tiles. The features must be stored in a numpy array with the shape (n_tiles, n_features), 
            be named `features.npy` and be stored in a subdirectory named after the `sample_ID`.
        PATH_COL_SIGNS : Path, optional
            Path to the file containing the column names of the molecular signatures. If not provided, only the Classic and Basal components will be used.
        PATH_TUM_ANNOT : Path, optional
            Path to the tumor annotation file. Used for the tumor segmentation task. The file must be stored in a subdirectory named after the `sample_ID` and be named `tiles_coord.npy`. 
            The file must contain the columns `z`, `k`, `x`, `y` and `tum`.
        col_name : str, optional
            Column name of the molecular signature to use. If not provided, only the Classic and Basal components will be used.
        ------

        """
        self.df = pd.read_csv(PATH_SUMMARY_DATA)
        try:
            self.df = self.df.dropna(
                subset=["histological_aspect", "sous_type_visuel", "stroma_visual_aspect"]
            )
        except KeyError:
            pass
        assert self.df.isna().any().any() == False, "There are NaN values in the dataset"
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
            - "custom": returns the component specified in the `col_name` attribute
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
        """
        Loads features for each slide.

        Parameters
        ----------
        n_tiles : int, default=10_000
            Number of tiles to sample per slide

        Returns
        -------
        features : np.ndarray
            Features of the tiles
        coordinates : np.ndarray
            Coordinates of the tiles
        slidenames : np.ndarray
            Names of the slides
        patient_ids : np.ndarray
            Patient ids


        """
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

            # random sampling of the tiles, if possible
            # Remark : this is rather slow because of random sampling inducing random access to the file
            # To disbale random sampling, just comment the following lines and uncomment the line above
            x_temp = np.load(row.path, mmap_mode="r")
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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute cross validation split for the dataset. Uses GroupKFold to ensure that the same patient is not present in both the train and validation set.
        
        Parameters
        ----------
        labels : pd.Series
            Series containing the labels of the dataset. The index must be the patient ids.
        use_cross_val : bool, default=True
            If True, use cross validation. If False, the train and validation set will be the same.
            
        Returns
        -------
        cv_train_ids : Dict[str, np.ndarray]
            Dictionary containing the indices of the training set for each split
        cv_val_ids : Dict[str, np.ndarray]
            Dictionary containing the indices of the validation set for each split    
        """

        groups = labels.index.values
        cv_train_ids, cv_val_ids = {}, {}
        if not use_cross_val:
            cv_train_ids["split_0"] = labels.index.values
            cv_val_ids["split_0"] = labels.index.values
            return cv_train_ids, cv_val_ids

        gkf = GroupKFold(n_splits=5)
        for i, (train_idx, val_idx) in enumerate(gkf.split(labels, labels, groups)):
            cv_train_ids[f"split_{i}"] = labels.iloc[train_idx].index.values.squeeze()
            cv_val_ids[f"split_{i}"] = labels.iloc[val_idx].index.values.squeeze()

        return cv_train_ids, cv_val_ids

    def _process_row(self, row, n_tiles):
        """
        Process a row of the dataframe to get the features and annotations of the tiles.

        Parameters
        ----------
        row : pd.Series
            Row of the dataframe. Must contain the columns `sample_ID`.

        Returns
        -------
        features : np.ndarray
            Features of the tiles
        annotations : np.ndarray
            Annotations of the tiles
        ids : list
            List of the patient ids
        """
        path_coord = self.PATH_TUM_ANNOT / row["sample_ID"] / "tiles_coord.npy"
        path_feat = self.PATH_FEATURES_DIR / row["sample_ID"] / "features.npy"

        coord = np.load(path_coord, allow_pickle=True).astype(int)[:, [0, 2, 3, 4]] # z,x,y,tum
        features = np.load(path_feat, allow_pickle=True, mmap_mode="r").astype(np.float32)

        # sample the data
        if features.shape[0] < n_tiles:
            print(
                f"Warning: {row['sample_ID']} has less than {n_tiles} tiles, only {len(features)} tiles will be used."
            )
        else:
            sampled_idx = np.random.choice(features.shape[0], n_tiles, replace=False)
            features = features[sampled_idx]
            # features = features[:, :n_tiles]

        mapped_feat, mapped_annot, mapped_ids = [], [], []
        for feat in features:
            z, x, y = feat[:3]
            idx = np.where((coord[:, 0] == z) & (coord[:, 1] == x) & (coord[:, 2] == y))[0]
            if len(idx) == 0:
                print(f"Warning: tile {z,x,y} not found in {row['sample_ID']}")
                raise ValueError
            else:
                mapped_feat.append(feat[3:])
                mapped_annot.append(coord[idx, 3]) # Ensures that the annotation is correctly mapped to the feature
                mapped_ids.append(row["sample_ID"])

        if len(mapped_feat) == 0:
            return None, None, None
        else:
            return np.array(mapped_feat), np.concatenate(mapped_annot, 0), mapped_ids

    def _get_rows_datas(self, rows, n_tiles, njobs=4):
        X, y, ids = [], [], []
        results = Parallel(n_jobs=njobs)(
            delayed(self._process_row)(row, n_tiles)
            for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Processing rows")
        )
        # results = [process_row(row, params) for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Processing rows")]
        for x, y_, id_ in results:
            if x is not None:
                X.append(x)
                y.append(y_)
                ids.extend(id_)
        return np.concatenate(X, 0), np.concatenate(y, 0), ids

    def get_data_tumors(self, n_tiles, njobs=4):
        """
        Get the data for the tumors segmentation task.
        We assume that the training data are given by the rows of the dataframe whose cohort
        is `DISC` and the validation data are given by the rows of the dataframe whose cohort
        is `BJN_U`.

        Parameters
        ----------
        n_tiles : int
            Number of tiles to sample per slide
        njobs : int
            Number of jobs to run in parallel

        Returns
        -------
        X_train : np.ndarray
            Features of the training set
        y_train : np.ndarray
        ids_train : list
            List of the patient ids of the training set

        X_val : np.ndarray
            Features of the validation set
        y_val : np.ndarray
        ids_val : list
            List of the patient ids of the validation set
        """

        rows_train = self.df[self.df.cohort == "DISC"]
        rows_val = self.df[self.df.cohort == "BJN_U"]

        X_train, y_train, ids_train = self._get_rows_datas(rows_train, n_tiles, njobs)
        X_val, y_val, ids_val = self._get_rows_datas(rows_val, n_tiles, njobs)

        return X_train, y_train, ids_train, X_val, y_val, ids_val
