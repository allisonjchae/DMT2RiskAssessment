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
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

from data.accession import AccessionConverter


class PMBBSample(NamedTuple):
    PMBB_ID: str
    data: Dict[str, Any]
    daterange: int


class PMBBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        accession_converter: AccessionConverter,
        filenames: Sequence[Union[Path, str]] = [],
        cache_path: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        identifier: str = "A1C",
        seed: int = int(time.time())
    ):
        """
        Args:
            accession_converter: a mapping from PMBB accession numbers to study
                dates.
            filenames: a sequence of filenames to the raw PMBB data sheets.
            cache_path: an optional path to a cache of previously loaded data
                for faster load times.
            verbose: optional flag for verbose stdout messages. Default True.
            identifier: specifies key for data points. If `A1C`, each A1C
                value is treated as a separate datapoint and patients can
                be represented multiple times. If `PMBB_ID`, only the most
                recent A1C value for a patient is used.
            seed: optional random seed. Default to seconds since epoch.
        """
        self.accession_converter = accession_converter
        self.filenames = filenames
        self.cache_path = cache_path
        self.identifier = identifier
        self.verbose = verbose
        self.pmbb_id_a1c_map = {}
        self.pmbb_ids = []
        self.rng = np.random.RandomState(seed)
        self.data = {}

        if self.cache_path is not None and os.path.isfile(self.cache_path):
            self._load_from_cache()
        else:
            self._manual_load()
            self.pmbb_id_a1c_map = self._make_trainable()
            self.pmbb_ids = np.array(list(self.pmbb_id_a1c_map.keys()))
            self.rng.shuffle(self.pmbb_ids)
            if self.cache_path is not None and self.cache_path != "None":
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
        self.identifier = cache["identifier"]
        if self.verbose != cache["verbose"] and self.verbose:
            print(
                f"Verbose flag updated: {self.verbose} to {cache['verbose']}"
            )
        self.verbose = cache["verbose"]
        self.data = cache["data"]
        self.pmbb_id_a1c_map = cache["pmbb_id_a1c_map"]
        self.pmbb_ids = cache["pmbb_ids"]
        self.accession_converter = cache["accession_converter"]
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

        return
        # For the imaging data, convert PMBB Accession numbers to study dates.
        imaging_keys = {
            "steatosis_run_2_merge.csv": "PMBB_ACCESSION",
            "visceral_merge_log_run_11_1_20.csv": "PENN_ACCESSION"
        }
        for key, val in imaging_keys.items():
            if key == "steatosis_run_2_merge.csv":
                self.data[key].PMBB_ACCESSION = self.data[
                    key
                ].PMBB_ACCESSION.apply(
                    self.accession_converter.get_date
                )
            elif key == "visceral_merge_log_fun_11_1_20.csv":
                self.data[key].PENN_ACCESSION = self.data[
                    key
                ].PENN_ACCESSION.apply(
                    self.accession_converter.get_date
                )
            self.data[key].rename(columns={val: "ENC_DATE_SHIFT"})

    def _make_trainable(self) -> Dict[str, pd.DataFrame]:
        """
        Takes a loaded dataset and gradually restricts it into a dataset that
        can actually be used for model training. This function performs the
        following operations:
            (1) Restricts each data type to only the rows with PMBB IDs of
                patients that have at least one complete entry in every data
                type in this PMBBDataset object.
            (2) Restricts each data type to just the row with the most recent
                entry for every PMBB ID.
        Input:
            None.
        Returns:
            A mapping of PMBB IDs to A1C values and dates.
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
        #     entry for every remaining PMBB ID. This step is only performed
        #     if self.identifier == `PMBB_ID`.
        if self.identifier.upper() == "PMBB_ID":
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
        pmbb_id_a1c_map = {}
        a1c_key = "PMBB_A1C_Deidentified_042020.csv"
        for pmbb_id in valid_pmbb_ids:
            pmbb_id_a1c_map[pmbb_id] = self.data[a1c_key][
                self.data[a1c_key].PMBB_ID == pmbb_id
            ][["RESULT_DATE_SHIFT", "RESULT_VALUE_NUM"]]
        return pmbb_id_a1c_map

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
        cache["pmbb_id_a1c_map"] = self.pmbb_id_a1c_map
        cache["pmbb_ids"] = self.pmbb_ids
        cache["identifier"] = self.identifier
        cache["columns"] = PMBBDataset._columns()
        cache["accession_converter"] = self.accession_converter
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
        return sum([df.shape[0] for _id, df in self.pmbb_id_a1c_map.items()])

    def __getitem__(self, idx: int) -> PMBBSample:
        """
        Retrieves a sample from the PMBB dataset.
        Input:
            idx: index of sample to retrieve.
            None.
        Returns:
            A PMBBSample object from the PMBB dataset.
        """
        if self.identifier.upper() == "PMBB_ID":
            pmbb_id = self.pmbb_ids[idx]
            data = {}
            min_time = None
            max_time = None
            a1c_date = None
            birth_date = None
            hispanic = False
            for key in self.data.keys():
                patient_data = self.data[key][
                    self.data[key]["PMBB_ID"] == pmbb_id
                ]
                for header in patient_data.columns.tolist():
                    if header == "RACE_HISPANIC_YN":
                        hispanic = hispanic or bool(
                            int(float(patient_data[header].tolist()[0]))
                        )
                    # Skip over PMBB IDs and PMBB Accession numbers.
                    if header in ["PMBB_ID", "PMBB_ACCESSION"]:
                        continue
                    # We also don't need normal range limits for A1C values.
                    if header in [
                        "VALUE_LOWER_LIMIT_NUM", "VALUE_UPPER_LIMIT_NUM"
                    ]:
                        continue
                    # Record the max time span of the sample data, and also
                    # calculate the age of the patient at the time of the A1C
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
        elif self.identifier.upper() == "A1C":
            counter = 0
            for pmbb_id in self.pmbb_ids:
                pmbb_id_count = self.pmbb_id_a1c_map[pmbb_id].shape[0]
                if counter + pmbb_id_count > idx:
                    break
                counter += pmbb_id_count
            a1c_date, a1c_val = list(self.pmbb_id_a1c_map[pmbb_id].to_numpy()[
                idx - counter
            ])
            data = {}
            data["RESULT_VALUE_NUM"] = a1c_val
            a1c_date = datetime.strptime(a1c_date, "%Y-%m-%d")
            min_time, max_time = a1c_date, a1c_date
            birth_date = None
            hispanic = False

            def nearest(items: Sequence[str], pivot: datetime) -> str:
                return datetime.strftime(
                    pd.to_datetime(
                        min(
                            [i for i in items],
                            key=lambda x: abs(
                                datetime.strptime(x, "%Y-%m-%d") - pivot
                            )
                        )
                    ),
                    "%Y-%m-%d"
                )

            for key in self.data.keys():
                patient_data = self.data[key][
                    self.data[key]["PMBB_ID"] == pmbb_id
                ]
                if "ENC_DATE_SHIFT" in patient_data.columns:
                    patient_data = patient_data[
                        patient_data.ENC_DATE_SHIFT == nearest(
                            patient_data.ENC_DATE_SHIFT.to_list(), a1c_date
                        )
                    ]
                elif "ENC_DT_SHIFT" in patient_data.columns:
                    patient_data = patient_data[
                        patient_data.ENC_DT_SHIFT == nearest(
                            patient_data.ENC_DT_SHIFT.to_list(), a1c_date
                        )
                    ]
                else:
                    patient_data = patient_data.sample(frac=1).reset_index()
                for header in patient_data.columns.tolist():
                    if header == "index":
                        continue
                    if header == "RACE_HISPANIC_YN":
                        hispanic = hispanic or bool(
                            int(float(patient_data[header].tolist()[0]))
                        )
                    # Skip over PMBB IDs and PMBB Accession numbers.
                    if header in ["PMBB_ID", "PMBB_ACCESSION"]:
                        continue
                    # We also don't need normal range limits for A1C values.
                    if header in [
                        "VALUE_LOWER_LIMIT_NUM", "VALUE_UPPER_LIMIT_NUM"
                    ]:
                        continue
                    # Record the max time span of the sample data, and also
                    # calculate the age of the patient at the time of the A1C
                    # result.
                    elif header in [
                        "ENC_DATE_SHIFT",
                        "ENC_DT_SHIFT",
                        "Birth_date_SHIFT"
                    ]:
                        data_date = datetime.strptime(
                            patient_data[header].tolist()[0], "%Y-%m-%d"
                        )
                        if header == "Birth_date_SHIFT":
                            birth_date = data_date
                            continue
                        if data_date < min_time:
                            min_time = data_date
                        if data_date > max_time:
                            max_time = data_date
                    else:
                        data[header] = patient_data[header].tolist()[0]

        data["AGE"] = a1c_date.year - birth_date.year - (
            (a1c_date.month, a1c_date.day) < (
                birth_date.month, birth_date.day
            )
        )
        if hispanic:
            data["RACE_CODE"] = "HISPANIC"
        data.pop("RACE_HISPANIC_YN", "")
        return PMBBSample(pmbb_id, data, (max_time - min_time).days)

    def to_tabular_partitions(
        self, partitions: Sequence[float] = [0.8, 0.1, 0.1]
    ) -> Tuple[pd.DataFrame]:
        """
        Convert the dataset to partitions of training, validation, and test
        sets in table format for compatibility with the PyTorch Tabular
        framework.
        Input:
            partitions: fractional paritions of training, validation, and test
                set sizes. Should sum to 1.0.
        Returns:
            Training, validation, and test dataset DataFrames.
        """
        if sum(partitions) != 1.0:
            raise ValueError(f"Partitions {partitions} do not sum to 1.0.")
        idxs = np.array(list(range(len(self))))
        self.rng.shuffle(idxs)
        train_cutoff = int(partitions[0] * len(self))
        val_cutoff = int((partitions[0] + partitions[1]) * len(self))

        train_table, val_table, test_table = [], [], []
        columns = list(self[0].data.keys())
        for i in idxs[:train_cutoff]:
            train_item_data = self[i].data
            ordered_data = []
            for key in columns:
                ordered_data.append(train_item_data[key])
            train_table.append(ordered_data)
        for j in idxs[train_cutoff:val_cutoff]:
            val_item_data = self[j].data
            ordered_data = []
            for key in columns:
                ordered_data.append(val_item_data[key])
            val_table.append(ordered_data)
        for k in idxs[val_cutoff:]:
            test_item_data = self[k].data
            ordered_data = []
            for key in columns:
                ordered_data.append(test_item_data[key])
            test_table.append(ordered_data)
        train_df = pd.DataFrame(train_table, columns=columns)
        val_df = pd.DataFrame(val_table, columns=columns)
        test_df = pd.DataFrame(test_table, columns=columns)
        return train_df, val_df, test_df

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

    @staticmethod
    def get_num_col_names(
        use_clinical: bool = True, use_idp: bool = True
    ) -> Sequence[str]:
        """
        Retrieves the column names of continuous data fields in the dataset.
        Input:
            use_clinical: whether to include clinical data names.
            use_idp: whether to include image-derived phenotype (IDP) data
                names.
        Returns:
            Column names of the appropriate continuous data fields.
        """
        clinical_features = [
            "AGE", "BP_SYSTOLIC", "BP_DIASTOLIC", "WEIGHT_LBS", "HEIGHT_INCHES"
        ]
        idp_features = [
            "LIVER_MEAN_HU",
            "SPLEEN_MEAN_HU",
            "VISCERAL_METRIC_AREA_MEAN",
            "SUBQ_METRIC_AREA_MEAN"
        ]
        features = []
        if use_clinical:
            features += clinical_features
        if use_idp:
            features += idp_features
        return features

    @staticmethod
    def get_cat_col_names(
        use_clinical: bool = True, use_idp: bool = True
    ) -> Sequence[str]:
        """
        Retrieves the column names of categorical data fields in the dataset.
        Input:
            use_clinical: whether to include clinical data names.
            use_idp: whether to include image-derived phenotype (IDP) data
                names.
        Returns:
            Column names of the appropriate categorical data fields.
        """
        clinical_features = ["RACE_CODE", "Sex"]
        idp_features = []
        features = []
        if use_clinical:
            features += clinical_features
        if use_idp:
            features += idp_features
        return features

    @staticmethod
    def get_target_col_name() -> Sequence[str]:
        """
        Retrieves the column names of output data field in the dataset.
        Input:
            None.
        Returns:
            Column names of the target output data field.
        """
        return ["RESULT_VALUE_NUM"]
