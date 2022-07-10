"""
PMBB vanilla data import tool.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Union


class PMBBDataset:
    def __init__(
        self,
        pkl_file: Optional[Union[Path, str]] = None,
        demographics: Union[Path, str] = None,
        phecodes: Union[Path, str] = None,
        race: Union[Path, str] = None,
        vitals: Union[Path, str] = None,
        biomarkers: Union[Path, str] = None,
        image1: Union[Path, str] = None,
        image2: Union[Path, str] = None,
        verbose: bool = False
    ):
        """
        Args:
            pkl_file: optional cache file to load dataset from.
            demographics: file path to the demographics dataset.
            phecodes: file path to the phecodes dataset.
            race: file path to the race dataset.
            vitals: file path to the vitals dataset.
            biomarkers: file path to the biomarkers dataset.
            image1: file path to the first image dataset.
            image2: file path to the second image dataset.
            verbose: file path to the verbose dataset.
        """
        self.verbose = verbose
        if pkl_file is not None:
            if not isinstance(pkl_file, Path):
                pkl_file = Path(pkl_file)
            if pkl_file.exists():
                loaded_data = pickle.load(open(pkl_file, "rb"))
                self.pmbb_ids = loaded_data["pmbb_ids"]
                self.data = loaded_data["data"]
                self.a1c = loaded_data["a1c"]
                self.t2dm = loaded_data["t2dm"]
                self.pkl_file = pkl_file
                if self.verbose:
                    print(
                        f"Imported from cache {os.path.abspath(pkl_file)}"
                    )
                return
            elif self.verbose:
                print(
                    f"Cache {pkl_file} not found, continuing with import."
                )
        self.demographics = demographics
        self.phecodes = phecodes
        self.race = race
        self.vitals = vitals
        self.biomarkers = biomarkers
        if not isinstance(self.biomarkers, Path):
            self.biomarkers = Path(self.biomarkers)
        self.image1 = image1
        self.image2 = image2
        self.pmbb_ids = set([])
        self.pkl_file = None
        self.t2dm = {}
        self.a1c = {}

        self.data = self._basic_import()
        self.data.update(self._biomarkers_import())
        self.data.update(self._image_import())

    def save_dataset(self, file_path: str) -> None:
        """
        Saves dataset and PMBB_IDs.
        Input:
            file_path: string file path to save the current dataset state to.
        Returns:
            None. File is generated according to specified path.
        """
        dataset_as_dict = {}
        dataset_as_dict["pmbb_ids"] = self.pmbb_ids
        dataset_as_dict["data"] = self.data
        dataset_as_dict["t2dm"] = self.t2dm
        dataset_as_dict["a1c"] = self.a1c
        pickle.dump(dataset_as_dict, open(file_path, "wb"))
        if self.verbose:
            print(f"Saved dataset to {os.path.abspath(file_path)}")
        return None

    def data(self) -> Dict[str, Any]:
        """
        Returns the current dataset.
        Input:
            None.
        Returns:
            The current dataset.
        """
        return self.data

    def exclude_by_missing_key(self, key: str) -> None:
        """
        Excludes patients that are missing specified key value.
        Input:
            key: key to use to exclude patients.
        Returns:
            None. self.data is directly modified.
        """
        if key is None or not isinstance(key, str) or len(key) == 0:
            if self.verbose:
                print(f"Key {key} not found in dataset. Skipping...")
            return
        key = key.lower()
        if key not in self.data.keys() and key not in ["a1c", "t2dm"]:
            if self.verbose:
                print(f"Key {key} not found in dataset. Skipping...")
            return
        # Get all the unique IDs from the specified dataset.
        # We assume that all PMBB_IDs in the key dataset have an associated
        # dataset value (ie no "nan" or "None" values).
        if key == "a1c":
            ref_dataset = self.a1c
        elif key == "t2dm":
            ref_dataset = self.t2dm
        else:
            ref_dataset = self.data[key]
        pmbb_id_idx = np.where(ref_dataset["header"] == "PMBB_ID")[0]
        assert pmbb_id_idx == 0
        valid_pmbb_ids = set(
            np.squeeze(ref_dataset["data"][:, pmbb_id_idx]).tolist()
        )

        self.pmbb_ids = self.pmbb_ids.intersection(valid_pmbb_ids)

        for k in self.data.keys():
            if k == key:
                continue
            datak = self.data[k]
            valid_idxs = np.zeros(
                datak["data"][:, pmbb_id_idx].shape, dtype=bool
            )
            before_size = np.squeeze(datak["data"][:, pmbb_id_idx]).shape[0]
            for _id in valid_pmbb_ids:
                valid_idxs[
                    np.where(datak["data"][:, pmbb_id_idx] == _id)
                ] = True
            datak["data"] = datak["data"][np.squeeze(valid_idxs), ...]
            after_size = np.squeeze(datak["data"][:, pmbb_id_idx]).shape[0]
            self.data[k] = datak
            if self.verbose:
                print(f"Key {k.upper()}: {after_size} / {before_size} kept.")
        if key != "a1c":
            valid_idxs = np.zeros(
                self.a1c["data"][:, pmbb_id_idx].shape, dtype=bool
            )
            before_size = np.squeeze(self.a1c["data"][:, pmbb_id_idx]).shape[0]
            for _id in valid_pmbb_ids:
                valid_idxs[
                    np.where(self.a1c["data"][:, pmbb_id_idx] == _id)
                ] = True
            self.a1c["data"] = self.a1c["data"][np.squeeze(valid_idxs), ...]
            after_size = np.squeeze(self.a1c["data"][:, pmbb_id_idx]).shape[0]
            if self.verbose:
                print(f"Key a1c: {after_size} / {before_size} kept.")
        if key != "t2dm":
            valid_idxs = np.zeros(
                self.t2dm["data"][:, pmbb_id_idx].shape, dtype=bool
            )
            before_size = np.squeeze(
                self.t2dm["data"][:, pmbb_id_idx]
            ).shape[0]
            for _id in valid_pmbb_ids:
                valid_idxs[
                    np.where(self.t2dm["data"][:, pmbb_id_idx] == _id)
                ] = True
            self.t2dm["data"] = self.t2dm["data"][np.squeeze(valid_idxs), ...]
            after_size = np.squeeze(self.t2dm["data"][:, pmbb_id_idx]).shape[0]
            if self.verbose:
                print(f"Key a1c: {after_size} / {before_size} kept.")

        if self.verbose:
            print()
        return

    def __len__(self) -> int:
        """
        Returns the number of unique PMBB_IDs in the datasets.
        Input:
            None.
        Returns:
            The number of unique PMBB_ID objects in the datasets.
        """
        return len(self.pmbb_ids)

    def get_unique_pmbb_ids(self) -> List[str]:
        """
        Returns a list of the unique PMBB_IDs in the dataset.
        Input:
            None.
        Returns:
            A list of the unique PMBB_IDs in the dataset.
        """
        return list(self.pmbb_ids)

    def gen_categorical_dimensions(
        self, data: np.ndarray, col: int = 1
    ) -> np.ndarray:
        """
        Convert a string categorical dimension into multiple binary dimensions.
        Input:
            data: an NxD data array, where N is the number of entries and D is
                the number of data dimensions.
            col: the column index of the string categorical dimension in data.
        Returns:
            An NxC binary mask, where C is the number of unique entries in
                the col column of data.
        """
        column = np.array(sorted(
            list(set(np.squeeze(data[:, :col]).astype(str).tolist()))
        ))

        res = np.empty((data.shape[0],) + column.shape)
        for i, el in enumerate(data[:, :col]):
            vec = np.zeros(column.shape).astype(np.float32)
            vec[np.where(column == el)] = 1.0
            res[i, :] = np.squeeze(vec[np.newaxis, ...])
        return res

    def _basic_import(self) -> Dict[str, Any]:
        """
        Imports demographics, race, vitals, and phecodes data.
        Input:
            None.
        Returns:
            A dictionary of file stems with associated imported data.
        """
        data = {}
        if self.pkl_file is not None:
            if self.verbose:
                print("Data already loaded from cache, skipping import...")
            return data

        demographics = None
        num_lines = 0
        with open(self.demographics, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1
                line = line[:-1] if line[-1] == "\n" else line
                if demographics is None:
                    demographics = []
                    header = re.split("\s+", line)
                    continue
                pmbb_id, sex, birth_date, age_at_enrollment = re.split(
                    "\s+", line
                )
                try:
                    demographics.append([
                        pmbb_id.upper(),
                        sex.lower(),
                        datetime.strptime(birth_date, "%Y-%m-%d"),
                        float(age_at_enrollment)
                    ])
                    self.pmbb_ids.add(pmbb_id.upper())
                except ValueError:
                    continue
        if self.verbose:
            print(
                f"Imported {len(demographics)} / {num_lines} patient " +
                "demographics."
            )
        data["demographics"] = {
            "header": np.array(header),
            "data": np.array(demographics)
        }

        race = None
        num_lines = 0
        with open(self.race, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1
                line = line[:-1] if line[-1] == "\n" else line
                if race is None:
                    race = []
                    header = re.split("\s+", line)
                    continue
                try:
                    pmbb_id, race_code, race_hispanic_yn = re.split(
                        "\s+", line
                    )
                    race.append([
                        pmbb_id.upper(),
                        race_code.lower(),
                        int(race_hispanic_yn),
                    ])
                    self.pmbb_ids.add(pmbb_id.upper())
                except ValueError:
                    continue
        if self.verbose:
            print(f"Imported {len(race)} / {num_lines} patient races.")
        data["race"] = {
            "header": np.array(header),
            "data": np.array(race)
        }

        vitals = None
        num_lines = 0
        with open(self.vitals, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1
                line = line[:-1] if line[-1] == "\n" else line
                if vitals is None:
                    vitals = []
                    header = ["PMBB_ID", "BMI", "OUTPATIENT"]
                    continue
                pmbb_id, _, _, _, bmi, _, visit_date = re.split("\s+", line)
                try:
                    vitals.append([
                        pmbb_id.upper(),
                        float(bmi),
                        datetime.strptime(visit_date, "%Y-%m-%d"),
                    ])
                    self.pmbb_ids.add(pmbb_id.upper())
                except ValueError:
                    continue
        if self.verbose:
            print(f"Imported {len(vitals)} / {num_lines} patient vitals.")
        data["vitals"] = {
            "header": np.array(header),
            "data": np.array(vitals)
        }

        phecodes = None
        num_lines = 0
        with open(self.phecodes, "r", encoding="utf-8") as f:
            for line in f:
                num_lines += 1
                line = line[:-1] if line[-1] == "\n" else line
                if phecodes is None:
                    phecodes = []
                    header = re.split("\s+", line)
                    continue
                line = re.split("\s+", line)
                pmbb_id, line = line[0], line[1:]
                line = [int(x) if x.isdigit() else x for x in line]
                try:
                    phecodes.append(line)
                    self.pmbb_ids.add(pmbb_id.upper())
                except ValueError:
                    continue
        if self.verbose:
            print(
                f"Imported {len(phecodes)} / {num_lines} patient PheCodes."
            )
        data["phecodes"] = {
            "header": np.array(header),
            "data": np.array(phecodes)
        }

        if self.verbose:
            print()
        return data

    def _biomarkers_import(self) -> Dict[str, Any]:
        """
        Imports biomarker data.
        Input:
            None.
        Returns:
            A dictionary of file stems with associated imported data.
        """
        data = {}
        if self.pkl_file is not None:
            if self.verbose:
                print("Data already loaded from cache, skipping import...")
            return data

        for fn in self.biomarkers.iterdir():
            # Skip any particular files.
            if "fasting_glucose" in fn.stem.lower():
                continue
            # Skip CRP since PMBB IDs are not available in CRP dataset.
            elif "crp" in fn.stem.lower() or "urobili" in fn.stem.lower():
                continue
            # Skip Urobili since PMBB IDs are not available in Urobili dataset.
            elif "urobili" in fn.stem.lower():
                continue
            # Skip ALB since PMBB IDs are not available in ALB dataset.
            elif "alb" in fn.stem.lower():
                continue
            elif not fn.as_posix().lower().endswith(".csv"):
                continue

            # Don't try to import or use the EMR data.
            if "emr" in fn.stem.lower():
                # Drop EMR data.
                if self.verbose:
                    print("Skipping " + fn.stem, flush=True)
                continue
            f = pd.read_csv(fn)
            f.dropna(inplace=True)
            if "bmi" in fn.stem.lower():
                f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            elif "dbp" in fn.stem.lower():
                f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            elif "sbp" in fn.stem.lower():
                f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            elif "demographics" in fn.stem.lower():
                # Don't drop anything for the demographics source data.
                pass
            elif "weight" in fn.stem.lower():
                f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            elif "height" in fn.stem.lower():
                f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            elif "alt" in fn.stem.lower():
                f.drop([
                    "PATIENT_CLASS", "ORDER_NAME", "ABNORMAL"
                ], axis=1, inplace=True)
            elif "t2dm_status" in fn.stem.lower():
                f.drop(["PACKET_UUID"], axis=1, inplace=True)
                f.replace("CONTROL", 0, inplace=True)
                f.replace("INSUFFICENT_EVIDENCE", 0, inplace=True)
                f.replace("CASE", 1, inplace=True)
                self.t2dm = {
                    "header": np.array(list(f.columns)),
                    "data": f.to_numpy()
                }
                if self.verbose:
                    print("Found ground truth T2DM statuses.")
                continue
            elif "antihypertensive" in fn.stem.lower():
                f.drop([
                    "PACKET_UUID", "GENO_ID", "PT_ID", "QUANTITY", "REFILLS"
                ], axis=1, inplace=True)
            elif "_asa_" in fn.stem.lower():
                f.drop([
                    "PACKET_UUID", "GENO_ID", "PT_ID", "QUANTITY", "REFILLS"
                ], axis=1, inplace=True)
            elif "smoking_history" in fn.stem.lower():
                # Only keeping the first, third, and last columns for smoking
                # history.
                f.drop([
                    "SOCIAL_HISTORY_TYPE",
                    "USE_ITEMS",
                    "USE_AMOUNT_PER_TIME",
                    "USE_YEARS",
                    "USE_QUIT_DATE_SHIFT"
                ], axis=1, inplace=True)
            elif "sc" in fn.stem.lower():
                f.drop(["PATIENT_CLASS", "ORDER_NAME"], axis=1, inplace=True)
            elif "medications" in fn.stem.lower():
                # Drop medication data.
                if self.verbose:
                    print("Skipping " + fn.stem, flush=True)
                continue
            elif "procedure" in fn.stem.lower():
                # Drop procedure data.
                if self.verbose:
                    print("Skipping " + fn.stem, flush=True)
                continue
            elif "diagnosis" in fn.stem.lower():
                # Only keeping the first two columns for diagnosis.
                pass
            else:
                f.drop([
                    "ORDER_NAME",
                    "LAB_RESULT_ITEM_DESCRIPTION",
                    "RESULT_DATE_SHIFT",
                    "ABNORMAL",
                ], axis=1, inplace=True)
                if "PATIENT_CLASS" in f.columns:
                    f.drop(["PATIENT_CLASS"], axis=1, inplace=True)
            if self.verbose:
                print("Imported " + fn.stem, flush=True)

            key = fn.stem.lower().split("_")
            key = "_".join(key[1:key.index("deidentified")])
            if "diagnosis" in fn.stem.lower():
                # Keep only the first two columns: PMBB_ID and CODE.
                data[key] = {
                    "header": np.array(list(f.columns))[:2],
                    "data": f.to_numpy()[:, :2]
                }
            elif "a1c" in fn.stem.lower():
                self.a1c = {
                    "header": np.array(list(f.columns)),
                    "data": f.to_numpy()
                }
                if self.verbose:
                    print("Found ground truth A1C measurements.")
                continue
            else:
                data[key] = {
                    "header": np.array(list(f.columns)),
                    "data": f.to_numpy()
                }
            self.pmbb_ids.update(f.to_numpy()[:, 0].tolist())

        return data

    def _image_import(self) -> Dict[str, Any]:
        """
        Imports image data.
        Input:
            None
        Returns:
            A dictionary of file stems with associated imported image data.
        """
        data = {}
        if self.pkl_file is not None:
            if self.verbose:
                print("Data already loaded from cache, skipping import...")
            return data

        if self.image1 is not None:
            if not isinstance(self.image1, Path):
                self.image1 = Path(self.image1)
            f = pd.read_csv(self.image1)
            f.drop([
                "PMBB_SERIES_UID",
                "THICKNESS",
                "NUM_IMAGES_STACK",
                "CONV_TOKEN",
                "NUM_SLICES_ABDOMEN",
                "Z_INDEX_LOW",
                "Z_INDEX_HIGH",
                "SUBQ_PIXEL_VOLUME",
                "SUBQ_METRIC_VOLUME",
                "VISCERAL_PIXEL_VOLUME",
                "VISCERAL_METRIC_VOLUME"
            ], axis=1, inplace=True)
            if self.verbose:
                print("Imported " + self.image1.stem, flush=True)
            key = self.image1.stem.lower().split("_")[0] + "_image_derived"
            data[key] = {
                "header": np.array(list(f.columns)),
                "data": f.to_numpy()
            }
            self.pmbb_ids.update(f.to_numpy()[:, 0].tolist())

        if self.image2 is not None:
            if not isinstance(self.image2, Path):
                self.image2 = Path(self.image2)
            f = pd.read_csv(self.image2)
            f.drop([
                "PMBB_SERIES_ID",
                "THICKNESS",
                "NUM_IMAGES",
                "CONV_TOKEN",
                "CHANCE_CONTRAST",
                "LIVER_METRIC_VOLUME",
                "LIVER_PIXEL_VOLUME",
                "LIVER_MEDIAN_HU",
                "SPLEEN_METRIC_VOLUME",
                "SPLEEN_PIXEL_VOLUME",
                "SPLEEN_MEDIAN_HU",
                "TIME_TO_PROCESS"
            ], axis=1, inplace=True)
            if self.verbose:
                print("Imported " + self.image2.stem, flush=True)
            key = self.image2.stem.lower().split("_")[0] + "_image_derived"
            data[key] = {
                "header": np.array(list(f.columns)),
                "data": f.to_numpy()
            }
            self.pmbb_ids.update(f.to_numpy()[:, 0].tolist())

        return data
