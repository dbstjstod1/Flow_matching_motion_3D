"""Check the public CQ500 case selection used for the manuscript."""
import json
from pathlib import Path

from .dataset_cq500 import index_cq500, select_series, split_patients


def selection(root):
    parts = split_patients(select_series(index_cq500(str(root))))
    # The paper evaluates the first 30 held-out records, not the entire remainder.
    parts["test"] = parts["test"][:30]
    return {split: [{"patient": row["patient"], "n_slices": row["n_slices"],
                     "thickness_mm": row["thickness_mm"]} for row in rows]
            for split, rows in parts.items()}


def check_dataset(root):
    expected = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/cq500_split.json").read_text())
    actual = selection(root)
    if actual != expected:
        raise ValueError("CQ500 selection differs from configs/cq500_split.json. "
                         "Use the complete DICOM dataset and rebuild a stale _cq500_index.json; "
                         "do not interpret shifted patient indices as the paper cohort.")
    print("Verified CQ500 series selection: 150 train / 50 validation / 30 test.")
