"""
Defines the PMBB-sourced dataset class.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import time
import torch
from typing import Any, Dict, NamedTuple, Optional, Sequence, Union


class PMBBSample(NamedTuple):
    PMBB_ID: str
    data: Dict[str, Any]
    daterange: int


class PMBBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames: Sequence[Union[Path, str]] = [],
        cache_path: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        seed: int = int(time.time())
    ):
        """
        Args:
            filenames: a sequence of filenames to the raw PMBB data sheets.
            cache_path: an optional path to a cache of previously loaded data
                for faster load times.
            verbose: optional flag for verbose stdout messages. Default True.
            seed: optional random seed. Default to seconds since epoch.
        """
        self.filenames = filenames
        self.cache_path = cache_path
        self.verbose = verbose
        self.pmbb_ids = []
        self.rng = np.random.RandomState(seed)
        self.data = {}

        if self.cache_path is not None and os.path.isfile(self.cache_path):
            self._load_from_cache()
        else:
            self._manual_load()
            self.pmbb_ids = self._make_trainable()
            if self.cache_path is not None:
                self._save_to_cache()

    def _load_from_cache(self) -> None:
        """
        Loads a dataset object from a cache file.
        Input:
            None.
        Returns:
            None. The fields of the dataset are modified directly.
        """
        with open(os.path.abspath(self.cache_path), "rb") as f:
            cache = pickle.load(f)
        self.filenames = cache["filenames"]
        if self.verbose != cache["verbose"] and self.verbose:
            print(
                f"Verbose flag updated: {self.verbose} to {cache['verbose']}"
            )
        self.verbose = cache["verbose"]
        self.data = cache["data"]
        self.pmbb_ids = cache["pmbb_ids"]
        if self.verbose:
            print(f"Loaded dataset from {os.path.abspath(self.cache_path)}")

    def _manual_load(self) -> None:
        """
        Loads a dataset object by parsing self.filenames.
        Input:
            None.
        Returns:
            None. The fields of the dataset are modified directly.
        """
        # Load the raw specified data files.
        for fn in self.filenames:
            # Assume that .txt file inputs are TSV files.
            if os.path.splitext(fn)[-1].lower() == ".txt":
                sep, header = "\t", 0
            # Otherwise, assume that the file is a CSV file.
            else:
                sep, header = ",", "infer"
            pmbb_data = pd.read_csv(
                os.path.abspath(fn), sep=sep, header=header
            )
            # Remove values from emergency or inpatient visits.
            if "PATIENT_CLASS" in pmbb_data.columns:
                pmbb_data = pmbb_data[
                    pmbb_data.PATIENT_CLASS == "OUTPATIENT"
                ]
            good_columns = PMBBDataset._columns()[os.path.basename(fn)]
            if len(good_columns) == 0 and self.verbose:
                print(
                    f"Warning: {os.path.abspath(fn)} has no specified columns."
                )
            self.data[os.path.basename(fn)] = pmbb_data[good_columns].dropna()

    def _make_trainable(self) -> Sequence[str]:
        """
        Takes a loaded dataset and gradually restricts it into a dataset that
        can actually be used for model training. This function performs the
        following operations:
            (1) Restricts each data type to only the rows with PMBB IDs of
                patients that have at least one complete entry in every data
                type in this PMBBDataset object.
            (2) Restricts each data type to just the row with the most recent
                entry for every PMBB ID.
            (3) Shuffle the order of the valid PMBB IDs based on the random
                seed.
        Input:
            None.
        Returns:
            A list of the remaining PMBB IDs after performing the above steps.
        """
        valid_pmbb_ids = None
        # Generate the intersection of all the PMBB IDs.
        for key in sorted(self.data, key=lambda x: self.data[x].shape[0]):
            if valid_pmbb_ids is None:
                valid_pmbb_ids = set(self.data[key].PMBB_ID.tolist())
                continue
            valid_pmbb_ids = valid_pmbb_ids & set(
                self.data[key].PMBB_ID.tolist()
            )
        # (1) Restrict each data type to only the rows with valid PMBB IDs.
        for key in self.data.keys():
            self.data[key] = self.data[key][
                self.data[key]["PMBB_ID"].isin(valid_pmbb_ids)
            ]
        # (2) Restrict each data type to just the row with the most recent
        #     entry for every remaining PMBB ID.
        for key in self.data.keys():
            if "ENC_DATE_SHIFT" in self._columns()[key]:
                sort_key = ["ENC_DATE_SHIFT"]
            elif "ENC_DT_SHIFT" in self._columns()[key]:
                sort_key = ["ENC_DT_SHIFT"]
            else:
                sort_key = None

            if sort_key is not None:
                self.data[key] = self.data[key].sort_values(
                    by=sort_key, ascending=False
                )
            self.data[key] = self.data[key].drop_duplicates(
                subset="PMBB_ID", keep="first"
            ).sort_index()
        # (3) Shuffle the order of the valid PMBB IDs.
        valid_pmbb_ids = np.array(list(valid_pmbb_ids))
        self.rng.shuffle(valid_pmbb_ids)
        return valid_pmbb_ids.tolist()

    def _save_to_cache(self) -> None:
        """
        Saves a dataset object to a specified cache file.
        Input:
            None.
        Returns:
            None.
        """
        cache = {}
        cache["filenames"] = self.filenames
        cache["verbose"] = self.verbose
        cache["data"] = self.data
        cache["pmbb_ids"] = self.pmbb_ids
        cache["columns"] = PMBBDataset._columns()
        with open(os.path.abspath(self.cache_path), "w+b") as f:
            pickle.dump(cache, f)
        if self.verbose:
            print(f"Saved dataset to {os.path.abspath(self.cache_path)}")

    def __len__(self) -> int:
        """
        Retrieves the length of the map-style dataset.
        Input:
            None.
        Returns:
            Length of the map-style dataset.
        """
        return len(self.pmbb_ids)

    def __getitem__(self, idx: int) -> PMBBSample:
        """
        Retrieves a sample from the PMBB dataset.
        Input:
            idx: index of sample to retrieve.
            None.
        Returns:
            A PMBBSample object from the PMBB dataset.
        """
        pmbb_id = self.pmbb_ids[idx]
        data = {}
        min_time = None
        max_time = None
        a1c_date = None
        birth_date = None
        for key in self.data.keys():
            patient_data = self.data[key][self.data[key]["PMBB_ID"] == pmbb_id]
            for header in patient_data.columns.tolist():
                # Skip over PMBB IDs and PMBB Accession numbers.
                if header in ["PMBB_ID", "PMBB_ACCESSION"]:
                    continue
                # We also don't need the normal range limits for A1c values.
                if header in [
                    "VALUE_LOWER_LIMIT_NUM", "VALUE_UPPER_LIMIT_NUM"
                ]:
                    continue
                # Record the max time span of the sample data, and also
                # calculate the age of the patient at the time of the A1c
                # result.
                elif header in [
                    "ENC_DATE_SHIFT",
                    "ENC_DT_SHIFT",
                    "RESULT_DATE_SHIFT",
                    "Birth_date_SHIFT"
                ]:
                    data_date = datetime.strptime(
                        patient_data[header].tolist()[0], "%Y-%m-%d"
                    )
                    if header == "Birth_date_SHIFT":
                        birth_date = data_date
                        continue
                    if min_time is None or data_date < min_time:
                        min_time = data_date
                    if max_time is None or data_date > max_time:
                        max_time = data_date
                    if header == "RESULT_DATE_SHIFT":
                        a1c_date = data_date
                else:
                    data[header] = patient_data[header].tolist()[0]
        data["AGE"] = a1c_date.year - birth_date.year - (
            (a1c_date.month, a1c_date.day) < (birth_date.month, birth_date.day)
        )
        return PMBBSample(pmbb_id, data, (max_time - min_time).days)

    @staticmethod
    def _columns() -> defaultdict:
        """
        Returns the columns to keep by PMBB filename.
        Input:
            None.
        Returns:
            A dictionary specifying the columns to keep within each PMBB data
            file.
        """
        columns = defaultdict(list)

        columns[
            "PMBB-Release-2020-2.2_phenotype_vitals-BMI-ct-studies.txt"
        ] = ["PMBB_ID", "BMI", "ENC_DATE_SHIFT"]
        columns[
            "PMBB-Release-2020-2.2_phenotype_race_eth-ct-studies.txt"
        ] = ["PMBB_ID", "RACE_CODE", "RACE_HISPANIC_YN"]
        columns[
            "PMBB-Release-2020-2.2_phenotype_demographics-ct-studies.txt"
        ] = ["PMBB_ID", "Sex", "Birth_date_SHIFT"]
        columns["PMBB_A1C_Deidentified_042020.csv"] = [
            "PMBB_ID",
            "RESULT_DATE_SHIFT",
            "RESULT_VALUE_NUM",
            "VALUE_LOWER_LIMIT_NUM",
            "VALUE_UPPER_LIMIT_NUM"
        ]
        columns["PMBB_SBP_Deidentified_042020.csv"] = [
            "PMBB_ID", "BP_SYSTOLIC", "ENC_DATE_SHIFT"
        ]
        columns["PMBB_DBP_Deidentified_042020.csv"] = [
            "PMBB_ID", "BP_DIASTOLIC", "ENC_DATE_SHIFT"
        ]
        columns["PMBB_Height_Deidentified_042020.csv"] = [
            "PMBB_ID", "HEIGHT_INCHES", "ENC_DATE_SHIFT"
        ]
        columns["PMBB_Weight_Deidentified_042020.csv"] = [
            "PMBB_ID", "WEIGHT_LBS", "ENC_DATE_SHIFT"
        ]
        columns["PMBB_Smoking_History_Deidentified_042020.csv"] = [
            "PMBB_ID", "SOCIAL_HISTORY_USE", "ENC_DT_SHIFT"
        ]
        columns["steatosis_run_2_merge.csv"] = [
            "PMBB_ID", "PMBB_ACCESSION", "LIVER_MEAN_HU", "SPLEEN_MEAN_HU"
        ]
        columns["visceral_merge_log_run_11_1_20.csv"] = [
            "PMBB_ID",
            "PMBB_ACCESSION",
            "VISCERAL_METRIC_AREA_MEAN",
            "SUBQ_METRIC_AREA_MEAN"
        ]

        return columns
