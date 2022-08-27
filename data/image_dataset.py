"""
Imaged-derived, PMBB-sourced dataset specifically adapted for A1C prediction.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Dict, NamedTuple, Optional, Sequence, Union

from data.dataset import PMBBDataset


class ImageSample(NamedTuple):
    pmbb_id: str
    subq_metric_area_mean: float
    visceral_metric_area_mean: float
    liver_mean_hu: float
    spleen_mean_hu: float
    biomarkers: Sequence[float]
    a1c_gt: float


class ImageDataset(Dataset):
    def __init__(
        self,
        steatosis_datapath: Union[Path, str],
        visceral_fat_datapath: Union[Path, str],
        biomarkers_datapath: Union[Path, str],
        biomarkers_keys: Sequence[str] = [],
        verbose: bool = False
    ):
        super().__init__()

        self.verbose = verbose
        self.visc_key = "visceral_image_derived"
        self.steat_key = "steatosis_image_derived"
        self.dataset = PMBBDataset(
            biomarkers=biomarkers_datapath,
            image1=visceral_fat_datapath,
            image2=steatosis_datapath,
            biomarkers_keys=biomarkers_keys,
            verbose=verbose
        )
        # self.dataset.exclude_by_missing_key("a1c")
        # Convert ICD-9 codes to T2DM status.
        self.dataset.extract_t2dm_from_icd9()

        valid_pmbb_ids = set([])
        from tqdm import tqdm
        for _id in tqdm(self.dataset.pmbb_ids):
            data_count = np.where(
                np.squeeze(
                    self.dataset.data[self.visc_key]["data"][:, 0]
                ) == _id
            )[0].shape
            data_count += np.where(
                np.squeeze(
                    self.dataset.data[self.steat_key]["data"][:, 0]
                ) == _id
            )[0].shape
            if data_count:
                valid_pmbb_ids.add(_id)
        # self.dataset.restrict_pmbb_ids(valid_pmbb_ids)
        self.valid_pmbb_ids = valid_pmbb_ids
        if self.verbose:
            print(f"PMBB ID Count: {len(self.dataset)}", flush=True)
            visc_count = self.dataset.data[self.visc_key]["data"].shape[0]
            print(f"Visceral Image Data Count: {visc_count}", flush=True)
            steat_count = self.dataset.data[self.steat_key]["data"].shape[0]
            print(f"Steatosis Image Data Count: {steat_count}\n", flush=True)

    def __len__(self) -> int:
        """
        Returns the number of samples in our dataset. Required for
        torch.utils.data.Dataset subclass implementation.
        Input:
            None.
        Returns:
            Number of samples in our dataset.
        """
        return len(self.dataset)

    def latest(
        self, time_interval: int = 90, image_before_a1c: bool = True
    ) -> None:
        """
        Restricts data to A1C data with associated image data within the
        specified time interval.
        Input:
            time_interval: maximum valid time between image data and A1C data
                collection dates in days.
            image_before_a1c: if True, then time_interval is sign-specific, ie
                a value of time_interval > 0 means that image data must have
                been acquired at most time_interval seconds before the A1C data
                (and not after the A1C data).
        Returns:
            None. Dataset is automatically modified.
        """
        a1c_pmbb_id_idx = np.where(
            self.dataset.a1c["header"] == "PMBB_ID"
        )[0][0]
        a1c_dates_idx = np.where(
            self.dataset.a1c["header"] == "ORDER_DATE_SHIFT"
        )[0][0]
        v_pmbb_id_idx = np.where(
            self.dataset.data[self.visc_key]["header"] == "PMBB_ID"
        )[0][0]
        v_dates_idx = None  # TODO
        s_pmbb_id_idx = np.where(
            self.dataset.data[self.steat_key]["header"] == "PMBB_ID"
        )[0][0]
        s_dates_idx = None  # TODO

        for i, date in enumerate(
            np.squeeze(self.dataset.a1c["data"][:, a1c_dates_idx])
        ):
            v_idxs = np.where(
                np.squeeze(
                    self.dataset.data[self.visc_key]["data"][:, v_pmbb_id_idx]
                ) == self.dataset.a1c["data"][i, a1c_pmbb_id_idx]
            )
            s_idxs = np.where(
                np.squeeze(
                    self.dataset.data[self.visc_key]["data"][:, s_pmbb_id_idx]
                ) == self.dataset.a1c["data"][i, a1c_pmbb_id_idx]
            )
            # TODO

    def map_keys(
        self,
        header: Union[pd.Index, list, np.ndarray],
        data: np.ndarray,
        mmap: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Maps a column from a dataset from one space into another target space.
        Input:
            header: description header of input dataset columns.
            dataset: input dataset.
            mmap: specified mapping pairs of [`key`, `val`] or [`val`, `key`].
        Returns:
            dataset with modified header and mapped column values.
        """
        if len(mmap.columns) != 2:
            raise AssertionError("Invalid map dimensions.")
        key, val = list(mmap.columns)
        mmap_key_idx = 0
        header = np.array(header)
        if val in header:
            key, val = val, key
            mmap_key_idx = -1
        mmap_np = mmap.to_numpy()

        key_idx = np.where(header == key)[0]
        for i in range(data.shape[0]):
            mapping_idx = np.where(
                np.squeeze(mmap_np[:, mmap_key_idx]) == data[i, key_idx]
            )[0]
            if len(mapping_idx) == 0:
                data[i, key_idx] = np.nan
                continue
            mapping_idx = mapping_idx[0]
            data[i, key_idx] = mmap_np[mapping_idx, mmap_key_idx + 1]
        data = data[~pd.isna(data).any(axis=1), :]
        header[key_idx] = val
        return {
            "header": header, "data": data
        }

    def pmbb_accessions_to_dates(self, mmap: Dict[str, Any]) -> None:
        """
        Converts PMBB accession numbers to datetime objects based on mmap.
        Input:
            mmap: mapping between PMBB accession numbers and datetime objects.
        Returns:
            None. Imaging datasets are directly modified (if present).
        """
        if self.visc_key in self.dataset.data:
            self.dataset.data[self.visc_key] = self.map_keys(
                self.dataset.data[self.visc_key]["header"],
                self.dataset.data[self.visc_key]["data"],
                pd.DataFrame(mmap["data"], columns=mmap["header"])
            )
        if self.steat_key in self.dataset.data:
            self.dataset.data[self.steat_key] = self.map_keys(
                self.dataset.data[self.steat_key]["header"],
                self.dataset.data[self.steat_key]["data"],
                pd.DataFrame(mmap["data"], columns=mmap["header"])
            )

    def import_study_dates(
        self,
        fn: Union[Path, str],
        mmap: Union[Path, str],
        modality: Optional[str] = None,
        deidentify: bool = False
    ) -> Dict[str, Any]:
        """
        Imports PMBB accession numbers and associated dates.
        Input:
            fn: file with Penn accession numbers and associated dates.
            mmap: file with mappings between Penn and PMBB accession numbers.
            modality: optional imaging modality to restrict dataset by.
            deidentify: optional flag to use deidentified PMBB accession
                numbers.
        Returns:
            A mapping between PMBB accession numbers and imaging dates.
        """
        df = pd.read_csv(fn)
        df.dropna(inplace=True)
        if modality is not None:
            df = df[df["STUDY_DESCRIPTION"].str.contains(modality, case=False)]
        df = df[["PENN_ACCESSION", "STUDY_DATE"]]

        # Map Penn Accession keys to PMBB Accession keys.
        df = self.map_keys(
            df.columns,
            df.to_numpy(),
            pd.read_csv(mmap, on_bad_lines="skip")
        )
        header, data = df["header"], df["data"]

        # Convert dates to datetime objects.
        dates_idx = np.where(header == "STUDY_DATE")[0][0]
        for i in range(data.shape[0]):
            data[i, dates_idx] = datetime.datetime.strptime(
                str(int(data[i, dates_idx])), "%Y%m%d"
            )

        # Deidentify accession key values.
        if deidentify:
            deidentification_zero_padding = 6
            accession_idx = np.where(
                np.array(header) == "PMBB_ACCESSION"
            )[0][0]
            for i in range(data.shape[0]):
                data[i, accession_idx] = int(
                    float(data[i, accession_idx]) // 10 ** (
                        deidentification_zero_padding
                    )
                ) * int(10 ** deidentification_zero_padding)

        return {
            "header": header, "data": data
        }

    def __getitem__(self, idx: int) -> ImageSample:
        """
        Loads and returns a sample from the dataset at the given index.
        Required for torch.utils.data.Dataset subclass implementation.
        Input:
            idx: index of desired dataset.
        Returns:
            An ImageSample object.
        """
        pmbb_id = self.dataset.get_unique_pmbb_ids()[idx]
        visceral_header = self.dataset.data[self.visc_key]["header"]
        visceral_data = self.dataset.data[self.visc_key]["data"]
        steatosis_header = self.dataset.data[self.steat_key]["header"]
        steatosis_data = self.dataset.data[self.steat_key]["data"]

        v_pmbb_idx = np.where(visceral_header == "PMBB_ID")[0][0]
        subq_idx = np.where(visceral_header == "SUBQ_METRIC_AREA_MEAN")[0][0]
        visc_idx = np.where(
            visceral_header == "VISCERAL_METRIC_AREA_MEAN"
        )[0][0]
        v_pmbb_id_idxs = np.where(
            np.squeeze(visceral_data[:, v_pmbb_idx]) == pmbb_id
        )[0]
        subq, visc = 0.0, 0.0
        for loc in v_pmbb_id_idxs:
            subq += visceral_data[loc, subq_idx]
            visc += visceral_data[loc, visc_idx]
        if len(v_pmbb_id_idxs):
            subq = subq / len(v_pmbb_id_idxs)
            visc = visc / len(v_pmbb_id_idxs)
        else:
            # If no data is available for this particular patient, just
            # substitute with the mean of the data that is actually available.
            subq = np.mean(visceral_data[:, subq_idx])
            visc = np.mean(visceral_data[:, visc_idx])

        s_pmbb_idx = np.where(steatosis_header == "PMBB_ID")[0][0]
        liver_mean_idx = np.where(steatosis_header == "LIVER_MEAN_HU")[0][0]
        spleen_mean_idx = np.where(steatosis_header == "SPLEEN_MEAN_HU")[0][0]
        s_pmbb_id_idxs = np.where(
            np.squeeze(steatosis_data[:, s_pmbb_idx]) == pmbb_id
        )[0]
        liver_mean, spleen_mean = 0.0, 0.0
        for loc in s_pmbb_id_idxs:
            liver_mean += steatosis_data[loc, liver_mean_idx]
            spleen_mean += steatosis_data[loc, spleen_mean_idx]
        if len(s_pmbb_id_idxs):
            liver_mean = liver_mean / len(s_pmbb_id_idxs)
            spleen_mean = spleen_mean / len(s_pmbb_id_idxs)
        else:
            # If no data is available for this particular patient, just
            # substitute with the mean of the data that is actually available.
            liver_mean = np.mean(steatosis_data[:, liver_mean_idx])
            spleen_mean = np.mean(steatosis_data[:, spleen_mean_idx])

        a1c_header = self.dataset.a1c["header"]
        a1c_data = self.dataset.a1c["data"]
        a1c_pmbb_idx = np.where(a1c_header == "PMBB_ID")[0][0]
        a1c_idx = np.where(a1c_header == "RESULT_VALUE_NUM")[0][0]
        a1c_pmbb_id_idxs = np.where(
            np.squeeze(a1c_data[:, a1c_pmbb_idx]) == pmbb_id
        )[0]
        a1c_mean = 0.0
        for loc in a1c_pmbb_id_idxs:
            a1c_mean += a1c_data[loc, a1c_idx]
        a1c_mean = a1c_mean / len(a1c_pmbb_id_idxs)

        biomarkers = []
        for k in sorted(self.dataset.data.keys()):
            k_pmbb_idx = np.where(
                self.dataset.data[k]["header"] == "PMBB_ID"
            )[0][0]
            # Handle BMI data.
            if "BMI" in self.dataset.data[k]["header"]:
                ref_key = "BMI"
            # Handle smoking history data.
            elif "SOCIAL_HISTORY_USE" in self.dataset.data[k]["header"]:
                ref_key = "SOCIAL_HISTORY_USE"
            # Handle age data.
            elif "AGE_042020" in self.dataset.data[k]["header"]:
                ref_key = "AGE_042020"
            else:
                continue
            k_data_idx = np.where(
                self.dataset.data[k]["header"] == ref_key
            )[0][0]
            k_pmbb_id_idxs = np.where(
                np.squeeze(
                    self.dataset.data[k]["data"][:, k_pmbb_idx]
                ) == pmbb_id
            )[0]
            k_mean = 0.0
            for loc in k_pmbb_id_idxs:
                # Handle smoking history separately.
                if "SOCIAL_HISTORY_USE" in self.dataset.data[k]["header"]:
                    k_mean += int(
                        self.dataset.data[k]["data"][
                            loc, k_data_idx
                        ].upper() != "NEVER"
                    )
                else:
                    k_mean += self.dataset.data[k]["data"][loc, k_data_idx]
            if len(k_pmbb_id_idxs):
                k_mean = k_mean / len(k_pmbb_id_idxs)
            else:
                # If not data is avilable for this particular patient, just
                # substitute with the mean of the data that is actually
                # available.
                k_mean = np.mean(self.dataset.data[k]["data"][:, k_data_idx])
            biomarkers.append(k_mean)

        return ImageSample(
            pmbb_id, subq, visc, liver_mean, spleen_mean, biomarkers, a1c_mean
        )
