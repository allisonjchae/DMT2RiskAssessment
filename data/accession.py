"""
Utility tool to convert PMBB accession numbers to study dates.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
from datetime import datetime
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union


class AccessionConverter:
    def __init__(
        self,
        pmbb_to_penn_fn: Union[Path, str],
        penn_to_date_fn: Union[Path, str]
    ):
        """
        Utility tool to convert PMBB accession numbers to study dates.
        Args:
            pmbb_to_penn_fn: CSV filepath containing the mappings from PMBB
                accession numbers to Penn accession numbers.
            penn_to_date_fn: CSV filepath containing the mappings from Penn
                accession numbers to study dates.
        """
        self.pmbb_to_penn_fn = pmbb_to_penn_fn
        self.pmbb_to_penn = pd.read_csv(
            os.path.abspath(pmbb_to_penn_fn),
            error_bad_lines=False,
            warn_bad_lines=False
        )
        self.penn_to_date_fn = penn_to_date_fn
        self.penn_to_date = pd.read_csv(os.path.abspath(penn_to_date_fn))
        self.penn_to_date = self.penn_to_date[
            ["PENN_ACCESSION", "STUDY_DATE"]
        ]

    def get_date(self, pmbb_accession_number: int) -> Optional[datetime]:
        """
        Returns the study date associated with a PMBB accession number.
        Input:
            pmbb_accession_number: A PMBB accession number.
        Returns:
            The associated study date represented as a datetime object.
        """
        penn_accession_number = self.pmbb_to_penn[
            self.pmbb_to_penn.PMBB_ACCESSION == int(pmbb_accession_number)
        ][["PENN_ACCESSION"]]
        if penn_accession_number.empty:
            return None
        penn_accession_number = penn_accession_number.iat[0, 0]
        study_date = self.penn_to_date[
            self.penn_to_date.PENN_ACCESSION == penn_accession_number
        ][["STUDY_DATE"]]
        if study_date.empty:
            return None
        return datetime.strptime(str(int(study_date.iat[0, 0])), "%Y%m%d")
