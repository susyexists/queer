"""Compatibility facade for queer.models.epw.EPWModel."""

import warnings
from pathlib import Path

import numpy as np

from .io.epw import read_epw_metadata
from .models.epw import EPWModel


class epw(EPWModel):
    def __init__(
        self,
        path=False,
        work_dir=False,
        outfolder=False,
        nk=False,
        nq=False,
        nph=False,
        ef=False,
        job_id=False,
    ):
        warnings.warn(
            "queer.epw is deprecated; use EPWModel.from_directory", DeprecationWarning, stacklevel=2
        )
        if path is False:
            if work_dir is False or outfolder is False:
                raise ValueError("supply path or both work_dir and outfolder")
            path = Path(work_dir) / outfolder
        self.path = Path(path).expanduser()
        self.work_dir, self.outfolder, self.job_id = work_dir, outfolder, job_id
        header = self.path / "gkk.out"
        values = read_epw_metadata(header) if header.is_file() else {}
        for key, value in (("nk", nk), ("nq", nq), ("nph", nph), ("ef", ef)):
            if value is not False and value is not None:
                values[key] = value
        for key in ("nk", "nq", "nph", "ef"):
            if key not in values:
                raise ValueError(f"EPW {key} must be supplied or available in gkk.out")
            setattr(self, key, values[key])
        self.q_path = np.arange(self.nq)
