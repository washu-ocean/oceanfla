from oceanfla.parser import parse_args
from pathlib import Path
# import cProfile


def main():

    parse_args()
    from oceanfla.config import all_opts, get_logger, finish_logging
    from nipype import config as ncfg
    from oceanfla.workflows import build_oceanfla_wf
    from oceanfla.utilities import clean_paths
    logger = get_logger("nipype.workflow")

    ncfg.update_config(
        {
            'execution': {
                'crashfile_format': "txt",
                'stop_on_first_crash': True,
            }
        }
    )   
    # nlogging.update_logging(ncfg)

    # print("building main workflow")
    oceanfla_wf = build_oceanfla_wf(
        subjects=all_opts.subject,
        base_dir=all_opts.work_dir
    )
    plugin_args_dict = {'n_procs': all_opts.n_procs, 'memory_gb': all_opts.mem_gb}

    # oceanfla_wf.config['execution'] = {
    #     'crashfile_format': "txt",
    #     'stop_on_first_crash': True,
    # }
    # oceanfla_wf.config['logging'] = {
    #     'log_directory': str(all_opts.log_dir.resolve()),
    #     'log_to_file': True
    # }

    # print("running main workflow")
    logger.info("starting oceanfla!")
    oceanfla_wf.run(plugin="MultiProc", plugin_args=plugin_args_dict)
    logger.info("oceanfla is finished!")
    
    if not all_opts.debug:
        clean_paths([
            Path(all_opts.raw_layout.connection_manager.database_file).parent,
            Path(all_opts.preproc_layout.connection_manager.database_file).parent,
            all_opts.work_dir
        ])
    finish_logging()
