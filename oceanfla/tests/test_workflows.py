import pytest
from oceanfla import workflows as workflows_module


@pytest.mark.parametrize(
    "opts", [
        {},
        {"session":None},
        {"fwhm":4},
        {"task":["oddball"]}
    ], 
    indirect=True
)
def test_build_oceanfla_wf_returns_workflow(
        monkeypatch, 
        tmp_path,
        opts,
        mock_global_logger
        ):

    monkeypatch.setattr(workflows_module, "all_opts", opts)
    wf = workflows_module.build_oceanfla_wf(subjects=opts.subject, base_dir=tmp_path)

    if opts.task != ["movie"]:
        assert wf is None
        mock_global_logger.warning.assert_any_call("NO WORKFLOWS WERE CREATED, EXITING NOW")
        mock_global_logger.warning.assert_any_call(f"NO BOLD RUNS FOUND FOR SUBJECT:1001, SESSION:01, TEMPLATE_SPACE:{opts.func_space}")
    else:
        assert wf is not None
        assert wf.name == f"oceanfla_task_{opts.task_rename}_wf"
        mock_global_logger.info.assert_any_call(f"creating the functional-space-level workflow: space_{opts.func_space}_wf")

    mock_global_logger.info.assert_any_call(f"creating the session-level workflow: sub_1001_ses_01_wf")
    
    

@pytest.mark.parametrize(
    "opts", [
        {},
        {"ignore":["event1"]},
        {"save_intermediates":True},
        {"repetition_time":1.5}
    ],
    indirect=True
)
def test_build_ses_design_wf_creates_workflow(monkeypatch, opts):
    monkeypatch.setattr(workflows_module, "all_opts", opts)
    wf = workflows_module.build_ses_design_wf(run="01", task="oddball")

    assert wf is not None
    assert "task_oddball_run_01_design_wf" in wf.name

    if opts.group or opts.ignore:
        assert wf.get_node('modify_events_file_node') is not None

    if opts.save_intermediates:
        assert wf.get_node("event_matrix_ds") is not None

    if not opts.repetition_time:
        assert wf.get_node(f"task_oddball_run_01_get_metadata_node") is not None
    else:
        assert wf.get_node("events_matrix_node").interface.inputs.tr == opts.repetition_time


def test_parse_session_bold_files_uses_fixture_layout(bids_layouts):
    layout = bids_layouts["preproc_layout"]
    run_info = workflows_module.parse_session_bold_files(
        layout=layout,
        subject="1001",
        session="01",
        tasks=["movie"],
    )

    assert ("MNI152NLin6Asym" in run_info) and ("MNIInfant" in run_info)
    assert "movie" in run_info["MNI152NLin6Asym"]
    assert len(run_info["MNI152NLin6Asym"]["movie"]) == 2

