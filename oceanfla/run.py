from nipype import config as ncfg
from oceanfla.parser import parse_args


# ncfg.update_config(
#     {
#         'execution': {
#             'crashfile_format': "txt",
#             'stop_on_first_crash': True,
#         },
#         'logging' : {
#             'log_directory'
#         }
#     }
# )

def main():

    parse_args()
    from oceanfla.config import all_opts
    from oceanfla.workflows import build_oceanfla_wf

    print("building main workflow")
    oceanfla_wf = build_oceanfla_wf(
        subjects=all_opts.subject,
        base_dir=all_opts.work_dir
    )
    plugin_args_dict = {'n_procs': all_opts.n_procs, 'memory_gb': all_opts.mem_gb}

    oceanfla_wf.config['execution'] = {
        'crashfile_format': "txt",
        'stop_on_first_crash': True,
    }
    print("running main workflow")
    oceanfla_wf.run(plugin="MultiProc", plugin_args=plugin_args_dict)
