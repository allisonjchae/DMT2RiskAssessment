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
from torch.utils.data import Dataset
from typing import NamedTuple, Union

from data.dataset import PMBBDataset


class ImageSample(NamedTuple):
    pmbb_id: str
    subq_metric_area_mean: float
    visceral_metric_area_mean: float
    liver_mean_hu: float
    spleen_mean_hu: float
    a1c_gt: float


class ImageDataset(Dataset):
    def __init__(
        self,
        steatosis_datapath: Union[Path, str],
        visceral_fat_datapath: Union[Path, str],
        biomarkers_datapath: Union[Path, str],
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
            restrict_to_image_data=True,
            verbose=verbose
        )
        self.dataset.exclude_by_missing_key("a1c")
        valid_pmbb_ids = set([])
        for _id in self.dataset.pmbb_ids:
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
        self.dataset.restrict_pmbb_ids(valid_pmbb_ids)
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
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
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

        return ImageSample(
            pmbb_id, subq, visc, liver_mean, spleen_mean, a1c_mean
        )
