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
        verbose: bool = False
    ):
        self.demographics = demographics
        self.phecodes = phecodes
        self.race = race
        self.vitals = vitals
        self.biomarkers = biomarkers
        self.verbose = verbose 
        self.pmbb_ids = set([])

        self.data = self._basic_import()

    def data(self) -> Dict[str, Any]:
        return self.data
    
    def __len__(self) -> int:
        return len(self.pmbb_ids)

    def _basic_import(self) -> Dict[str, Any]:
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
                except ValueError:
                    continue
                self.pmbb_ids.add(pmbb_id.upper())
        if self.verbose:
            print(
                f"Imported {len(demographics)} / {num_lines} patients for " +
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
                except ValueError:
                    continue
                self.pmbb_ids.add(pmbb_id.upper())
        if self.verbose:
            print(f"Imported {len(race)} / {num_lines} patients for race.")
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
                except ValueError:
                    continue
                self.pmbb_ids.add(pmbb_id.upper())
        if self.verbose:
            print(f"Imported {len(vitals)} / {num_lines} patients for vitals.")
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
                pmbb_id = line[0].upper()
                line = [
                    int(x) if x.isdigit() else x for x in line[1:] 
                ]
                phecodes.append(line)
                self.pmbb_ids.add(pmbb_id)
        if self.verbose:
            print(
                f"Imported {len(phecodes)} / {num_lines} patients for PheCodes."
            )
        data["phecodes"] = {
            "header": np.array(header),
            "data": np.array(phecodes)
        }

        if self.verbose:
            print()
        return data


if __name__ == "__main__":
    PREFIX = os.path.join(
        os.path.abspath(os.path.join(__file__, os.pardir)),
        "PMBB-Release-2020-2.2_"
    )
    dataset = PMBBDataset(
        demographics=Path(PREFIX + "phenotype_demographics-ct-studies.txt"),
        phecodes=Path(PREFIX + "phenotype_PheCode-matrix-ct-studies.txt"),
        race=Path(PREFIX + "phenotype_race_eth-ct-studies.txt"),
        vitals=Path(PREFIX + "phenotype_vitals-BMI-ct-studies.txt"),
        biomarkers=None,
        verbose=True
    )
    print(len(dataset))