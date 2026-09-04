import pytest
from oceanfla import parser as parser_module

def test_parse_args_valid_minimum_configuration(monkeypatch, dummy_bids_layout):

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
        

def test_parse_args_rejects_missing_required_model(dummy_bids_layout):

    parser_inputs = [
        "--task", "oddball",
        "--derivs_dir", str(dummy_bids_layout["preproc_root"].parent),
        "--raw_bids", str(dummy_bids_layout["raw_root"]),
        "--work_dir", str(dummy_bids_layout["work_dir"]),
    ]
    with pytest.raises(SystemExit):
        parsed_args, parser, config_args = parser_module._build_parser(parser_inputs)