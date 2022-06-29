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
import re
from typing import Any, Dict, Union


class PMBBDataset:
    def __init__(
        self,
        demographics: Union[Path, str],
        phecodes: Union[Path, str],
        race: Union[Path, str],
        vitals: Union[Path, str],
        biomarkers: Union[Path, str],
        image1: Union[Path, str] = None,
        image2: Union[Path, str] = None,
        verbose: bool = False
    ):
        """
        Args:
            demographics: file path to the demographics dataset.
            phecodes: file path to the phecodes dataset.
            race: file path to the race dataset.
            vitals: file path to the vitals dataset.
            biomarkers: file path to the biomarkers dataset.
            image1: file path to the first image dataset.
            image2: file path to the second image dataset.
            verbose: file path to the verbose dataset.
        """
        self.demographics = demographics
        self.phecodes = phecodes
        self.race = race
        self.vitals = vitals
        self.biomarkers = biomarkers
        self.image1 = image1
        self.image2 = image2
        self.verbose = verbose
        self.pmbb_ids = set([])

        self.data = self._basic_import()
        self.data.update(self._biomarkers_import())
        self.data.update(self._image_import())

    def data(self) -> Dict[str, Any]:
        return self.data

    def __len__(self) -> int:
        """
        Returns the number of unique PMBB_IDs in the datasets.
        Input:
            None.
        Returns:
            The number of unique PMBB_ID objects in the datasets.
        """
        return len(self.pmbb_ids)

    def _basic_import(self) -> Dict[str, Any]:
        """
        Imports demographics, race, vitals, and phecodes data.
        Input:
            None.
        Returns:
            A dictionary of file stems with associated imported data.
        """
        data = {}

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

        for fn in self.biomarkers.iterdir():
            # Skip any particular files.
            if "fasting_glucose" in fn.stem.lower():
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
                # Drop T2DM status data.
                if self.verbose:
                    print("Skipping " + fn.stem, flush=True)
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


if __name__ == "__main__":
    PREFIX = os.path.join(
        os.path.abspath(os.path.join(__file__, os.pardir)),
        "PMBB-Release-2020-2.2_"
    )
    BIOMARKERS = os.path.join(
        os.path.abspath(os.path.join(__file__, os.pardir)),
        "biomarker_data"
    )
    IMAGE_DATA = os.path.join(
        os.path.abspath(os.path.join(__file__, os.pardir)),
        "image_data"
    )
    dataset = PMBBDataset(
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
