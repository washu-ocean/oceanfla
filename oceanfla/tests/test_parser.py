import pytest
from oceanfla import parser as parser_module

def test_parse_args_valid_minimum_configuration(monkeypatch, bids_layouts):

    parser_inputs = [
        "--task", "oddball",
        "--derivs_dir", str(bids_layouts["preproc_root"].parent),
        "--raw_bids", str(bids_layouts["raw_root"]),
        "--work_dir", str(bids_layouts["work_dir"]),
        "--func_space", "fsLR",
        "--hrf", "8", "16",
    ]
    parsed_args, parser, config_args = parser_module._build_parser(parser_inputs)
    monkeypatch.setattr(parser_module, "_build_parser", lambda arg_list=None : (parsed_args, parser, config_args))
    args = parser_module.parse_args()

    assert args.task == ["oddball"]
    assert args.derivs_dir == bids_layouts["preproc_root"].parent
    assert args.raw_bids == bids_layouts["raw_root"]
    assert args.work_dir == bids_layouts["work_dir"]
    assert args.task_rename == "oddball"
    assert args.preproc_bids == bids_layouts["preproc_root"]
        

def test_parse_args_rejects_missing_required_model(bids_layouts):

    parser_inputs = [
        "--task", "oddball",
        "--derivs_dir", str(bids_layouts["preproc_root"].parent),
        "--raw_bids", str(bids_layouts["raw_root"]),
        "--work_dir", str(bids_layouts["work_dir"]),
    ]
    with pytest.raises(SystemExit):
        parsed_args, parser, config_args = parser_module._build_parser(parser_inputs)