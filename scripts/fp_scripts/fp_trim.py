import argparse
import numpy as np
import os

import toast
from toast.utils import Logger
from astropy.table import QTable


FULL_DETS = "FULL"


def parse_requested_dets(requested_dets):
    if isinstance(requested_dets, str) and requested_dets.upper() == FULL_DETS:
        return FULL_DETS

    try:
        ndets = int(requested_dets)
    except (TypeError, ValueError):
        raise ValueError("requested_dets must be an even integer >= 10 or FULL.")

    if ndets < 10 or ndets % 2 != 0:
        raise ValueError("requested_dets must be an even integer >= 10 or FULL.")
    return ndets


def selected_module_table(dettable_full, module_arrays):
    if module_arrays == 1:
        wafer_slots = ["w2"]
    elif module_arrays == 3:
        wafer_slots = ["w1", "w2", "w3"]
    else:
        raise ValueError("module_arrays must be either 1 or 3.")

    wafer_mask = np.zeros(len(dettable_full), dtype=bool)
    for wafer_slot in wafer_slots:
        wafer_mask |= dettable_full['wafer_slot'] == wafer_slot
    return dettable_full[wafer_mask], wafer_slots


def generate_trimmed_table(module_dettable, ndets_selected):
    pairs_select = ndets_selected // 2

    pair_keys = []
    seen_pair_keys = set()
    for wafer_slot, pixel in zip(module_dettable['wafer_slot'], module_dettable['pixel']):
        key = (wafer_slot, pixel)
        if key in seen_pair_keys:
            continue
        seen_pair_keys.add(key)
        pair_keys.append(key)

    selected_pair_indices = np.linspace(0, len(pair_keys) - 1, pairs_select, dtype=int)
    selected_pair_keys = {pair_keys[index] for index in selected_pair_indices}

    mask = np.array(
        [
            (wafer_slot, pixel) in selected_pair_keys
            for wafer_slot, pixel in zip(module_dettable['wafer_slot'], module_dettable['pixel'])
        ]
    )
    trim_dettable = module_dettable[mask]

    if len(trim_dettable) != ndets_selected:
        raise RuntimeError(
            f"Selected {len(trim_dettable)} detectors, expected {ndets_selected}. "
            "Check the detector table pair structure."
        )

    return trim_dettable


def build_fp_file(requested_dets, module_arrays=1, fp_dir="./input_files/fp_files/"):
    """Build or reuse the focalplane file and return (ndets_selected, path)."""
    hf_fulltable_file = os.path.join(fp_dir, "fp_f280_dettable.h5")
    dettable_full = QTable.read(hf_fulltable_file, path='dettable_stack')

    requested_dets = parse_requested_dets(requested_dets)
    module_dettable, wafer_slots = selected_module_table(dettable_full, module_arrays)
    total_module_dets = len(module_dettable)

    if requested_dets == FULL_DETS:
        ndets_selected = total_module_dets
        trim_dettable = module_dettable
    else:
        ndets_selected = requested_dets
        if ndets_selected > total_module_dets:
            message = (
                f"Requested dets ({ndets_selected}) exceeds the total number of detectors "
                f"in {', '.join(wafer_slots)} ({total_module_dets} dets)."
            )
            if module_arrays == 1:
                message += " Try --module-arrays 3 or use --dets FULL."
            else:
                message += " Use --dets FULL to select all available detectors."
            raise ValueError(message)

        trim_dettable = generate_trimmed_table(module_dettable, ndets_selected)

    focalplane_file = f"dets_FP_PC280_{ndets_selected}.h5"
    fp_filename = os.path.join(fp_dir, focalplane_file)

    trim_dettable.meta["module_arrays"] = module_arrays
    trim_dettable.meta["requested_dets"] = str(requested_dets)
    trim_dettable.meta["wafer_slots"] = ",".join(wafer_slots)

    write_focalplane = True
    if os.path.exists(fp_filename):
        try:
            existing_dettable = QTable.read(fp_filename, path='dettable_trim')
            write_focalplane = (
                len(existing_dettable) != ndets_selected
                or existing_dettable.meta.get("module_arrays") != module_arrays
                or existing_dettable.meta.get("requested_dets") != str(requested_dets)
                or existing_dettable.meta.get("wafer_slots") != ",".join(wafer_slots)
            )
        except Exception:
            write_focalplane = True

    if write_focalplane:
        trim_dettable.write(
            fp_filename,
            path='dettable_trim',
            serialize_meta=True,
            overwrite=True,
        )

    return ndets_selected, fp_filename


def main():
    parser = argparse.ArgumentParser(description="Build a PrimeCam focalplane file.")
    parser.add_argument(
        "requested_dets",
        help="Number of detectors to select, or FULL for no trimming.",
    )
    parser.add_argument(
        "--module-arrays",
        "--MODULE_ARRAYS",
        dest="module_arrays",
        type=int,
        choices=[1, 3],
        default=int(os.environ.get("MODULE_ARRAYS", 1)),
        help="Number of module arrays to use: 1 selects w2, 3 selects w1/w2/w3.",
    )
    args = parser.parse_args()
    ndets_selected, fp_filename = build_fp_file(
        args.requested_dets,
        module_arrays=args.module_arrays,
    )
    Logger.get().info(
        f"Using {args.module_arrays} module array(s), focalplane file "
        f"{fp_filename} with {ndets_selected} detectors."
    )


if __name__ == "__main__":
    main()
