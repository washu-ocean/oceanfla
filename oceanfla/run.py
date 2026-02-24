from oceanfla.parser import parse_args
from nipype import config as ncfg
# import cProfile


def main():

    parse_args()
    from oceanfla.config import all_opts, get_logger, finish_logging, close_layouts
    from oceanfla.workflows import build_oceanfla_wf
    from oceanfla.utilities import clean_paths
    logger = get_logger("nipype.workflow")

    ncfg.update_config(
        {
            'execution': {
                'crashfile_format': "txt",
                'stop_on_first_crash': True,
                'crashdump_dir': all_opts.log_dir
            }
        }
    )   

    # Build and run the main workflow
    oceanfla_wf = build_oceanfla_wf(
        subjects=all_opts.subject,
        base_dir=all_opts.work
    )
    plugin_args_dict = {'n_procs': all_opts.n_procs, 'memory_gb': all_opts.mem_gb}

    logger.info("starting oceanfla!")
    if oceanfla_wf:
        oceanfla_wf.run(plugin="MultiProc", plugin_args=plugin_args_dict)
    logger.info("oceanfla is finished!")

    # Some clean-up
    close_layouts()
    if not all_opts.keep_work:
        clean_paths([
            all_opts.work
        ])
    finish_logging()
    
