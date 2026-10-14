import pytest
from types import SimpleNamespace
from oceanfla import workflows as workflows_module


@pytest.mark.parametrize(
    "opts", [
        {},
        {"session":None},
        {"fwhm":4, "percent_change":True},
        {"highpass":0.08, "lowpass":0.15},
        {"highpass":0.08, "lowpass":0.15, "nuisance_regression":["some_confound"]},
        {"nuisance_regression":["some_confound"]},
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

    assert wf.get_node("make_tmask_node") is not None
    assert wf.get_node("events_matrix_node") is not None
    assert wf.get_node("nuisance_matrix_node") is not None
    assert wf.get_node("main_design_ds") is not None
    assert wf.get_node("tmask_ds") is not None

    if opts.group:
        assert wf.get_node('modify_events_file_node') is not None

    if opts.continuous_vars:
        assert wf.get_node("find_continuous_vars_node") is not None



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


@pytest.mark.parametrize(
    "opts", [
        {},
        {"session":None},
        {"fwhm":4, "percent_change":True},
        {"highpass":0.08, "lowpass":0.15},
        {"highpass":0.08, "lowpass":0.15, "nuisance_regression":["some_confound"]},
        {"nuisance_regression":["some_confound"]},
        {"task":["oddball"]}
    ], 
    indirect=True
)
def test_build_run_workflow_uses_filter_output_when_filtering(monkeypatch, opts):

    monkeypatch.setattr(workflows_module, "all_opts", opts)

    wf = workflows_module.build_run_workflow(
        run=1,
        task="movie",
        func_space="MNI152NLin6Asym",
        file_extension=".nii.gz",
    )

    assert wf.get_node("final_outputnode") is not None


    if not opts.save_intermediates:
        if opts.highpass or opts.lowpass:
            assert wf.get_node("filtering_node") is not None
            assert wf._graph.get_edge_data(wf.get_node("filtering_node"), wf.get_node("final_outputnode")) is not None
        elif opts.nuisance_regression:
            assert wf.get_node("outputnode") is not None
            assert wf.get_node("glm_regression_node") is not None
            assert wf._graph.get_edge_data(wf.get_node("outputnode"), wf.get_node("final_outputnode")) is not None
        elif opts.percent_change:
            assert wf.get_node("percent_change_node") is not None
            assert wf._graph.get_edge_data(wf.get_node("percent_change_node"), wf.get_node("final_outputnode")) is not None
        elif opts.fwhm:
            assert wf.get_node("outputnode") is not None
            assert wf.get_node("volume_smooth_node") is not None
            assert wf._graph.get_edge_data(wf.get_node("outputnode"), wf.get_node("final_outputnode")) is not None