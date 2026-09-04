from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from unittest.mock import MagicMock
from bids import BIDSLayout
from bids.layout import BIDSLayoutIndexer

from oceanfla import config as config_module
from oceanfla import workflows as workflows_module

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
        "bold_path": bold_path,
        "mask_path": mask_path,
    }


@pytest.fixture
def all_opts_fixture(monkeypatch, tmp_path, dummy_bids_layout):
    opts = SimpleNamespace(
        task=["oddball"],
        task_rename="oddball",
        subject=["01"],
        session=None,
        func_space="MNI152NLin6Asym",
        preproc_layout=dummy_bids_layout["preproc_layout"],
        raw_layout=dummy_bids_layout["raw_layout"],
        preproc_bids=dummy_bids_layout["preproc_root"],
        raw_bids=dummy_bids_layout["raw_root"],
        brain_mask=str(dummy_bids_layout["mask_path"]),
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
        datasink_path=tmp_path / "output",
        bids_patterns=[],
        save_intermediates=False,
        filter_padtype="mean",
        filter_padlen=50,
        work_dir=dummy_bids_layout["work_dir"],
        work=dummy_bids_layout["work_dir"],
        debug=False,
        highpass=None,
        lowpass=None,
        fd_censoring=False,
        spike_regression=False,
        min_run_frames=0,
        run_exclusion_threshold=0.0,
        min_average_tsnr=0.0,
        percent_change=False,
        layouts=[dummy_bids_layout["preproc_layout"], dummy_bids_layout["raw_layout"]],
        exclusion_file=None,
        fwhm=None,
        stdscale_glm="sesLevel",
    )

    monkeypatch.setattr(config_module, "all_opts", opts)
    monkeypatch.setattr(workflows_module, "all_opts", opts)
    return opts


@pytest.fixture
def dummy_bids_layout(tmp_path, dummy_nifti):
    raw_root = tmp_path / "raw_bids"
    preproc_root = tmp_path / "derivatives" / "fmriprep"

    raw_root.mkdir(parents=True, exist_ok=True)
    preproc_root.mkdir(parents=True, exist_ok=True)

    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)

    (raw_root / "dataset_description.json").write_text(
        '{"Name": "dummy_raw_bids", "BIDSVersion": "1.8.0"}'
    )
    (preproc_root / "dataset_description.json").write_text(
        '{"Name": "dummy_preproc_bids", "BIDSVersion": "1.8.0", "GeneratedBy": [{"Name": "dummy"}]}'
    )

    raw_func_dir = raw_root / "sub-01" / "ses-01" / "func"
    preproc_func_dir = preproc_root / "sub-01" / "ses-01" / "func"
    raw_func_dir.mkdir(parents=True)
    preproc_func_dir.mkdir(parents=True)

    bold_file = raw_func_dir / "sub-01_ses-01_task-oddball_run-01_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), bold_file)
    raw_json = raw_func_dir / "sub-01_ses-01_task-oddball_run-01_bold.json"
    raw_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')

    bold_file = raw_func_dir / "sub-01_ses-01_task-oddball_run-02_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), bold_file)
    raw_json = raw_func_dir / "sub-01_ses-01_task-oddball_run-02_bold.json"
    raw_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')

    proc_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-01_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), proc_file)
    preproc_json = preproc_func_dir / "sub-01_ses-01_task-oddball_run-01_space-MNI152NLin6Asym_desc-preproc_bold.json"
    preproc_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')
    proc_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-02_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), proc_file)
    preproc_json = preproc_func_dir / "sub-01_ses-01_task-oddball_run-02_space-MNI152NLin6Asym_desc-preproc_bold.json"
    preproc_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')

    proc_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-01_space-MNIInfant_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), proc_file)
    preproc_json = preproc_func_dir / "sub-01_ses-01_task-oddball_run-01_space-MNIInfant_desc-preproc_bold.json"
    preproc_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')
    proc_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-02_space-MNIInfant_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(dummy_nifti["data"], np.eye(4)), proc_file)
    preproc_json = preproc_func_dir / "sub-01_ses-01_task-oddball_run-02_space-MNIInfant_desc-preproc_bold.json"
    preproc_json.write_text('{"RepetitionTime": 2.0, "TaskName": "oddball"}')

    events_file = raw_func_dir / "sub-01_ses-01_task-oddball_run-01_events.tsv"
    events_file.write_text("onset\tduration\ttrial_type\n0\t2\toddball\n")
    events_file = raw_func_dir / "sub-01_ses-01_task-oddball_run-02_events.tsv"
    events_file.write_text("onset\tduration\ttrial_type\n0\t2\toddball\n")

    confounds_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-01_desc-confounds_timeseries.tsv"
    confounds_file.write_text("framewise_displacement\ttrans_x\n0.0\t0.1\n")
    confounds_file = preproc_func_dir / "sub-01_ses-01_task-oddball_run-02_desc-confounds_timeseries.tsv"
    confounds_file.write_text("framewise_displacement\ttrans_x\n0.0\t0.1\n")

    mask_path = preproc_root / "sub-01" / "ses-01" / "func" / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((10, 11, 12), dtype=np.uint8), np.eye(4)), mask_path)

    raw_bids_db_path = work_dir / f".raw_indexer"
    raw_layout = BIDSLayout(
        root=str(raw_root),
        database_path=raw_bids_db_path,
        validate=False,
        indexer=BIDSLayoutIndexer(index_metadata=False),
    )
    preproc_bids_db_path = work_dir / f".preproc_indexer"
    preproc_layout = BIDSLayout(
        root=str(preproc_root),
        database_path=preproc_bids_db_path,
        validate=False,
        is_derivative=True,
        indexer=BIDSLayoutIndexer(index_metadata=False),
    )

    return {
        "raw_root": raw_root,
        "preproc_root": preproc_root,
        "raw_layout": raw_layout,
        "preproc_layout": preproc_layout,
        "work_dir":work_dir,
        "bold_file": proc_file,
        "events_file": events_file,
        "confounds_file": confounds_file,
        "raw_bold_file": bold_file,
        "mask_path": mask_path,
    }
