from nipype import config as ncfg
from oceanfla.parser import parse_args


ncfg.update_config(
    {
        'execution': {
            'crashfile_format': "txt",
            'stop_on_first_crash': True,
        }
    }
)

def main():

    work_dir = "/Users/agardr/Desktop/python_code/test_bids_data/work"
    parse_args()
    from .config import all_opts
    from .workflows import build_oceanfla_wf

    oceanfla_wf = build_oceanfla_wf(
        subjects=all_opts.subject,
        base_dir=work_dir
    )

    oceanfla_wf.run()