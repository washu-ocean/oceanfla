import pytest
from oceanfla.tests.conftest import make_all_opts
from oceanfla import workflows as workflows_module


@pytest.mark.parametrize("opts", [
    make_all_opts(),
    make_all_opts(session=None),
    make_all_opts(fwhm=4)

])
def test_build_oceanfla_wf_returns_workflow(
        monkeypatch, 
        tmp_path, 
        opts
        ):

    monkeypatch.setattr(workflows_module, "all_opts", opts)
    wf = workflows_module.build_oceanfla_wf(subjects=opts.subject, base_dir=tmp_path)

    assert wf is not None
    assert wf.name == "oceanfla_task_oddball_wf"
    

@pytest.mark.parametrize("opts", [make_all_opts()])
def test_build_ses_design_wf_creates_workflow(monkeypatch, opts):
    monkeypatch.setattr(workflows_module, "all_opts", opts)
    wf = workflows_module.build_ses_design_wf(run="01", task="oddball")

    assert wf is not None
    assert "task_oddball_run_01_design_wf" in wf.name


def test_parse_session_bold_files_uses_fixture_layout(dummy_bids_layout):
    layout = dummy_bids_layout["preproc_layout"]
    run_info = workflows_module.parse_session_bold_files(
        layout=layout,
        subject="01",
        session="01",
        tasks=["oddball"],
    )

    assert ("MNI152NLin6Asym" in run_info) and ("MNIInfant" in run_info)
    assert "oddball" in run_info["MNI152NLin6Asym"]
    assert len(run_info["MNI152NLin6Asym"]["oddball"]) == 2


def test_dummy_bids_layout_has_expected_files(dummy_bids_layout):
    assert dummy_bids_layout["bold_file"].exists()
    assert dummy_bids_layout["events_file"].exists()
    assert dummy_bids_layout["confounds_file"].exists()
