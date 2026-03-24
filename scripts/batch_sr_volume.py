#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import os
import tempfile
import glob

import nibabel as nib
import numpy as np
from os.path import join

from sr_volume import process_volume_sinsr

def main():
    data_dir = "/home/francois/Projects/data/raw_data/LnRobo/CT_Fluoro/240313/anatomical_variations"

    files = [
        "outpaint_result_sample00_20260210130410.nii.gz",
        "outpaint_result_sample01_20260210131145.nii.gz",
        "outpaint_result_sample02_20260210131909.nii.gz",
        "outpaint_result_sample03_20260210132646.nii.gz",
        "outpaint_result_sample04_20260210133424.nii.gz",
        "outpaint_result_sample05_20260210134207.nii.gz",
        "outpaint_result_sample06_20260210134954.nii.gz",
        "outpaint_result_sample07_20260210135741.nii.gz",
        "outpaint_result_sample08_20260210140535.nii.gz",
        "outpaint_result_sample09_20260210141339.nii.gz"
    ]

    target_files = [join(data_dir, f) for f in files]

    for file_path in target_files:
        print(f"Processing: {file_path}")
        output_name = file_path.replace(".nii.gz", "_sr.nii.gz")
        
        process_volume_sinsr(
            volume_path=file_path,
            output_path=output_name,
            sf=2,
            chop_size=128,
            chop_stride=112
        )

if __name__ == "__main__":
    main()