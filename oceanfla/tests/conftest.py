from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from unittest.mock import MagicMock
from bids import BIDSLayout
from bids.layout import BIDSLayoutIndexer
from niworkflows.utils.testing import generate_bids_skeleton
import shutil

from oceanfla import config as config_module
from oceanfla import workflows as workflows_module

PREPROC_LAYOUT = {
    "1001": [
        {
            "session":ses,
            "func": [
                *(
                    {
                        "task": "movie",
                        "run": i, 
                        "space": space,
                        "suffix": "bold",
                        "metadata": {
                            "RepetitionTime": 2.0, 
                            "TaskName": "movie"
                        }
                    }
                    for space in ["MNI152NLin6Asym", "MNIInfant"]
                    for i in range(2)
                ) 
            ],
            "anat": [
                *(
                    {
                        "hemi": hemi,
                        "space": "fsLR",
                        "suffix": "midthickness",
                        "extension": ".surf.gii"
                    }
                    for hemi in ["L" "R"]
                )
            ]
        }
        for ses in ["01", "02"]
    ],
    "1002": "*"
}

RAW_LAYOUT = {
    "1001": [
        {
            "session":ses,
            "func": [
                *(
                    {
                        "task": "movie",
                        "run": i, 
                        "suffix": "bold",
                        "metadata": {
                            "RepetitionTime": 2.0, 
                            "TaskName": "movie"
                        }
                    }
                    for i in range(2)
                ),
                *(
                    {
                        "task": "movie",
                        "run": i,
                        "suffix": "events",
                        "extension": ".tsv"
                    }
                    for i in range(2)
                )
            ]
        }
        for ses in ["01", "02"]
    ],
    "1002": "*"
}

@pytest.fixture(autouse=True)
def mock_global_logger(monkeypatch):
    """Disable the multiprocessing file logger used by config.py during tests."""
    mock_logger = MagicMock()
    mock_process = MagicMock()

    monkeypatch.setattr(config_module, "config_logging_process", lambda *args, **kwargs: (mock_process, MagicMock()))
    monkeypatch.setattr(config_module, "get_logger", lambda name, q=None: mock_logger)
    monkeypatch.setattr(workflows_module, "get_logger", lambda name, q=None: mock_logger)

    yield mock_logger

@pytest.fixture
def dummy_nifti(tmp_path):
    data = np.random.default_rng(0).normal(size=(10, 11, 12, 8)).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    bold_path = tmp_path / "dummy_bold.nii.gz"
    nib.save(img, bold_path)

    mask_data = np.ones((10, 11, 12), dtype=np.uint8)
    mask_path = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), mask_path)

    return {
        "data": data,
        "bold_img": img,
        "bold_path": bold_path,
        "mask_path": mask_path,
    }


@pytest.fixture(scope="session")
def bids_layouts(tmp_path_factory):

    base = tmp_path_factory.mktemp("dataset")
    raw_bids_dir = base / "rawdata"
    preproc_bids_dir = base / "derivatives" / "fmriprep"
    generate_bids_skeleton(raw_bids_dir, RAW_LAYOUT)
    generate_bids_skeleton(preproc_bids_dir, PREPROC_LAYOUT)

    # for bold_file in raw_bids_dir.glob("**/sub*.nii.gz"):
    #     nib.save(dummy_nifti["img"], bold_file)
    
    work_dir = base / "work"
    work_dir.mkdir(exist_ok=True)

    raw_bids_db_path = work_dir / f".raw_indexer"
    raw_layout = BIDSLayout(
        root=str(raw_bids_dir),
        database_path=raw_bids_db_path,
        validate=False,
        indexer=BIDSLayoutIndexer(index_metadata=False),
    )
    preproc_bids_db_path = work_dir / f".preproc_indexer"
    preproc_layout = BIDSLayout(
        root=str(preproc_bids_dir),
        database_path=preproc_bids_db_path,
        validate=False,
        is_derivative=True,
        indexer=BIDSLayoutIndexer(index_metadata=False),
    )
    return {
        "raw_root": raw_bids_dir,
        "preproc_root": preproc_bids_dir,
        "raw_layout": raw_layout,
        "preproc_layout": preproc_layout,
        "work_dir":work_dir,
    }

@pytest.fixture
def opts(request, bids_layouts, dummy_nifti):
    opts = SimpleNamespace(
        task=["movie"],
        task_rename="movie",
        subject=["1001", "1002"],
        session="01",
        func_space="MNI152NLin6Asym",
        preproc_layout=bids_layouts["preproc_layout"],
        raw_layout=bids_layouts["raw_layout"],
        preproc_bids=bids_layouts["preproc_root"],
        raw_bids=bids_layouts["raw_root"],
        brain_mask=str(dummy_nifti["mask_path"]),
        fd_threshold=0.5,
        minimum_unmasked_neighbors=0,
        start_censoring=0,
        dscans_path=None,
        fir=None,
        hrf=[8, 16],
        fir_vars=None,
        hrf_vars=None,
        unmodeled=None,
        parametric_modulators=None,
        group=None,
        ignore=None,
        repetition_time=2.0,
        confounds=[],
        exclude_run_mean=False,
        exclude_run_trend=False,
        nuisance_regression=[],
        generic_nuisance_columns=[],
        volterra_lag=None,
        volterra_columns=[],
        datasink_path=bids_layouts["preproc_root"],
        bids_patterns=[],
        save_intermediates=False,
        filter_padtype="mean",
        filter_padlen=50,
        work_dir=bids_layouts["work_dir"],
        work=bids_layouts["work_dir"],
        debug=False,
        highpass=None,
        lowpass=None,
        fd_censoring=False,
        spike_regression=False,
        min_run_frames=0,
        run_exclusion_threshold=0.0,
        min_average_tsnr=0.0,
        percent_change=False,
        layouts=[bids_layouts["preproc_layout"], bids_layouts["raw_layout"]],
        exclusion_file=None,
        fwhm=None,
        stdscale_glm="sesLevel",
        n_procs=4,
        mem_gb=5
    )

    kwargs = request.param
    for k, v in kwargs.items():
        setattr(opts, k, v)

    return opts

