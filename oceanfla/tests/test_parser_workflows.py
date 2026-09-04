import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
from bids import BIDSLayout
from nipype import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

from oceanfla import parser as parser_module
from oceanfla import workflows as workflows_module


class TestParser:

    def test_parse_args_valid_minimum_configuration(self, monkeypatch, tmp_path, dummy_bids_layout):
        parser_inputs = [
            "--task", "oddball",
            "--derivs_dir", str(dummy_bids_layout["preproc_root"].parent),
            "--raw_bids", str(dummy_bids_layout["raw_root"]),
            "--work_dir", str(dummy_bids_layout["work_dir"]),
            "--func_space", "fsLR",
            "--hrf", "8", "16",
        ]
        parsed_args, parser, config_args = parser_module._build_parser(parser_inputs)
        monkeypatch.setattr(parser_module, "_build_parser", lambda arg_list=None : (parsed_args, parser, config_args))
        args = parser_module.parse_args()

        assert args.task == ["oddball"]
        assert args.derivs_dir == dummy_bids_layout["preproc_root"].parent
        assert args.raw_bids == dummy_bids_layout["raw_root"]
        assert args.work_dir == dummy_bids_layout["work_dir"]
        assert args.task_rename == "oddball"
        assert args.preproc_bids == dummy_bids_layout["preproc_root"]
        assert args.work.exists()

    def test_parse_args_rejects_missing_required_model(self, monkeypatch, dummy_bids_layout, tmp_path):
        parser_inputs = [
            "--task", "oddball",
            "--derivs_dir", str(dummy_bids_layout["preproc_root"].parent),
            "--raw_bids", str(dummy_bids_layout["raw_root"]),
            "--work_dir", str(dummy_bids_layout["work_dir"]),
        ]
        with pytest.raises(SystemExit):
            parsed_args, parser, config_args = parser_module._build_parser(parser_inputs)


class TestWorkflows:
    def test_build_oceanfla_wf_returns_workflow(self, monkeypatch, tmp_path, all_opts_fixture):
        # dummy_workflow = Workflow(name="dummy_session_wf")
        inputnode = Node(IdentityInterface(fields=["task"]), name="inputnode")
        # dummy_workflow.add_nodes([inputnode])

        # monkeypatch.setattr(workflows_module, "build_session_wf", lambda subject, session=None: dummy_workflow)

        wf = workflows_module.build_oceanfla_wf(subjects=all_opts_fixture.subject, base_dir=tmp_path)

        assert wf is not None
        assert wf.name == "oceanfla_task_oddball_wf"

    def test_build_ses_design_wf_creates_workflow(self, all_opts_fixture):
        wf = workflows_module.build_ses_design_wf(run="01", task="oddball")

        assert wf is not None
        assert "task_oddball_run_01_design_wf" in wf.name

    def test_parse_session_bold_files_uses_fixture_layout(self, dummy_bids_layout):
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

    def test_dummy_bids_layout_has_expected_files(self, dummy_bids_layout):
        assert dummy_bids_layout["bold_file"].exists()
        assert dummy_bids_layout["events_file"].exists()
        assert dummy_bids_layout["confounds_file"].exists()
