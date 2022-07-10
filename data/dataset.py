"""
PMBB vanilla data import tool.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
from collections import defaultdict
from datetime import datetime, date
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Union


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
        restrict_to_image_data: bool = True,
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
            restrict_to_image_data: whether to only import image data.
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
        if self.biomarkers is not None and not isinstance(
            self.biomarkers, Path
        ):
            self.biomarkers = Path(self.biomarkers)
        self.image1 = image1
        self.image2 = image2
        self.pmbb_ids = set([])
        self.pkl_file = None
        self.t2dm = {}
        self.a1c = {}

        if restrict_to_image_data:
            self.data = self._image_import()
            self._biomarkers_import(import_key="a1c")
            assert len(self.a1c)
            return
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

    def get_range_timepoints(self, key: str) -> Dict[str, List[datetime]]:
        """
        Returns a range of the time for which specified data was collected
        for each patient in the specified dataset.
        Input:
            key: clinical data to get the timepoints for.
        Returns:
            dictionary keys: PMBB IDs.
            dictionary values: [earliest time that data was collected,
                most recent time that data was collected].
        """
        timepoints = {}
        if key == "a1c":
            dataset = self.a1c
        elif key == "t2dm":
            dataset = self.t2dm
        else:
            dataset = self.data[key]
        dates = dataset["header"].tolist().index("ORDER_DATE_SHIFT")
        ids = dataset["header"].tolist().index("PMBB_ID")
        for r in range(dataset["data"].shape[0]):
            curr_range = timepoints.get(
                dataset["data"][r, ids].upper(), [date.max, date.min]
            )
            d = datetime.strptime(dataset["data"][r, dates], "%Y-%m-%d")
            if d < datetime.combine(curr_range[0], datetime.min.time()):
                curr_range[0] = d
            if d > datetime.combine(curr_range[-1], datetime.min.time()):
                curr_range[-1] = d
            timepoints[dataset["data"][r, ids].upper()] = curr_range

        return timepoints

    def exclude_by_data_availability(
        self,
        timepoints: Dict[str, List[datetime]],
        threshmin: Union[float, int] = 1
    ) -> None:
        """
        Excludes patients that are missing too many types of data in
        specified time frame.
        Input:
            timepoints: a dictionary of PMBB IDs mapping to ranges of
                datetimes where data was collected.
            threshmin: threshold for number of data types the patient
                must have in order to remain in the dataset. If
                threshmin < 1, threshmin is treated as a fraction of
                the total number of available datatypes.
        Returns:
            None. self.data is directly modified.
        """
        if threshmin < 1.0:
            threshmin = int(threshmin * len(self.data.keys()))
        else:
            threshmin = int(threshmin)
        threshmin = max(0, min(threshmin, len(self.data.keys())))
        counts = defaultdict(int)
        for key in self.data.keys():
            ids = self.data[key]["header"].tolist().index("PMBB_ID")
            if "ORDER_DATE_SHIFT" in self.data[key]["header"].tolist():
                date_idx = self.data[key]["header"].tolist().index(
                    "ORDER_DATE_SHIFT"
                )
            elif "ENC_DATE_SHIFT" in self.data[key]["header"].tolist():
                date_idx = self.data[key]["header"].tolist().index(
                    "ENC_DATE_SHIFT"
                )
            else:
                date_idx = -1
            for i, el in enumerate(
                set(np.squeeze(self.data[key]["data"][:, ids]).tolist())
            ):
                if date_idx == -1:
                    counts[el] += 1
                    continue
                data_date = datetime.strptime(
                    self.data[key]["data"][i, date_idx],
                    "%Y-%m-%d"
                )
                l_date, r_date = timepoints[el]
                if l_date.year <= data_date.year <= r_date.year:
                    counts[el] += 1
        valid_pmbb_ids = []
        for _id in counts.keys():
            if counts[_id] > threshmin:
                valid_pmbb_ids.append(_id)
        self.restrict_pmbb_ids(set(valid_pmbb_ids))

    def restrict_pmbb_ids(
        self, valid_pmbb_ids: Set[str], key: Optional[str] = None
    ) -> None:
        """
        Excludes PMBB IDs and data in dataset to just those specified in
        valid_pmbb_ids.
        Input:
            valid_pmbb_ids: a collection of valid PMBB IDs to restrict our
                dataset to.
            key: optional original key used for dataset exclusion criteria.
        Returns:
            None. Dataset is automatically updated.
        """
        self.pmbb_ids = self.pmbb_ids.intersection(valid_pmbb_ids)
        pmbb_id_idx = 0

        for k in self.data.keys():
            if key is not None and k == key:
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
                print(
                    f"Key {k.upper()}: {after_size} / {before_size} kept.",
                    flush=True
                )
        if (key is not None and key != "a1c" and
                self.a1c.get("data", None) is not None):
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
                print(
                    f"Key a1c: {after_size} / {before_size} kept.", flush=True
                )
        if (key is not None and key != "t2dm" and
                self.t2dm.get("data", None) is not None):
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
                print(
                    f"Key a1c: {after_size} / {before_size} kept.", flush=True
                )

        if self.verbose:
            print()

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
                print(
                    f"Key {key} not found in dataset. Skipping...", flush=True
                )
            return
        key = key.lower()
        if key not in self.data.keys() and key not in ["a1c", "t2dm"]:
            if self.verbose:
                print(
                    f"Key {key} not found in dataset. Skipping...", flush=True
                )
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

        self.restrict_pmbb_ids(valid_pmbb_ids, key=key)

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

    def X(self) -> np.ndarray:
        print(sorted(self.data.keys()))
        data_classes = sorted(self.data.keys())
        for i in range(len(self.get_unique_pmbb_ids())):
            data_for_id = []
            _id = self.get_unique_pmbb_ids()[i]
            for j in range(len(data_classes)):
                d = self.data[data_classes[j]]
                numeric = "RESULT_VALUE_NUM" in d["header"]
                numeric = numeric and "VALUE_LOWER_LIMIT_NUM" in d["header"]
                numeric = numeric and "VALUE_UPPER_LIMIT_NUM" in d["header"]
                if numeric:
                    # Assume a normal distribution.
                    val_idx = np.where(
                        d["header"] == "RESULT_VALUE_NUM"
                    )[0][0]
                    lower_idx = np.where(
                        d["header"] == "VALUE_LOWER_LIMIT_NUM"
                    )[0][0]
                    upper_idx = np.where(
                        d["header"] == "VALUE_UPPER_LIMIT_NUM"
                    )[0][0]
                    pmbb_idx = np.where(
                        d["header"] == "PMBB_ID"
                    )[0][0]
                    data_rows = np.where(d["data"][:, pmbb_idx] == _id)[0]
                    if len(data_rows) == 0:
                        data_for_id.append(None)
                    else:
                        sum_z_val = 0
                        for i in range(len(data_rows)):
                            # Row with the relevant data.
                            row = np.squeeze(d["data"][data_rows[i]])
                            mean = row[lower_idx] + (
                                (row[upper_idx] - row[lower_idx]) / 2
                            )
                            std = (mean - row[lower_idx]) / 2
                            if std != 0:
                                sum_z_val += (row[val_idx] - mean) / std
                            else:
                                # This case should only be reached if these
                                # are percentages.
                                sum_z_val += row[val_idx] / 100
                        data_for_id.append(sum_z_val / len(data_rows))

                else:
                    if i == 0:
                        print(
                            data_classes[j],
                            self.data[data_classes[j]]["header"]
                        )

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

    def _biomarkers_import(
        self, import_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Imports biomarker data.
        Input:
            import_key: if specified, only this data type is imported.
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
            if import_key is not None and import_key not in fn.stem.lower():
                continue
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
            f.dropna(inplace=True)
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
            f.dropna(inplace=True)
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


if __name__ == "__main__":
    DATA_DIR = "./data"
    PREFIX = os.path.join(
        os.path.abspath(DATA_DIR),
        "PMBB-Release-2020-2.2_"
    )
    BIOMARKERS = os.path.join(
        os.path.abspath(DATA_DIR),
        "biomarker_data"
    )
    IMAGE_DATA = os.path.join(
        os.path.abspath(DATA_DIR),
        "image_data"
    )
    dataset = PMBBDataset(
        pkl_file=Path("./analysis/cache_file_a1c_filtered_30p_threshmin.pkl"),
        demographics=Path(PREFIX + "phenotype_demographics-ct-studies.txt"),
        phecodes=Path(PREFIX + "phenotype_PheCode-matrix-ct-studies.txt"),
        race=Path(PREFIX + "phenotype_race_eth-ct-studies.txt"),
        vitals=Path(PREFIX + "phenotype_vitals-BMI-ct-studies.txt"),
        biomarkers=Path(BIOMARKERS),
        image1=Path(
            os.path.join(IMAGE_DATA, "visceral_merge_log_run_11_1_20.csv")
        ),
        image2=Path(os.path.join(IMAGE_DATA, "steatosis_run_2_merge.csv")),
        verbose=True
    )
    dataset.X()
