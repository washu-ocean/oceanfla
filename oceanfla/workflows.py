from ast import mod
from nipype import Node, Workflow, Function
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import IdentityInterface, Select
from nipype.interfaces.workbench.cifti import CiftiSmooth
from niworkflows.utils.bids import collect_participants
from oceanfla.interfaces.utility import FLADataSink, ReadMetadataFile
from oceanfla.interfaces.clean import FilterData, PercentChange
from oceanfla.interfaces.events import EventsMatrix, GetVolumeCount, ModifyEventsFile
from oceanfla.interfaces.exclusions import CheckRunRetention, CheckRuntSNR
from oceanfla.interfaces.nuisance import GenerateNuisanceMatrix
from oceanfla.interfaces.regression import ConcatRegressionData, RunGLMRegression
from oceanfla.interfaces.tmask import MakeTmask, MakeTmaskTsv
from oceanfla.interfaces.utility import MergeUnique, ExtractDataGroup
from oceanfla.config import all_opts, get_logger
from oceanfla.interfaces.workbench_utils import CiftiParcellate, VolumeSmooth
from oceanfla.utilities import is_cifti_file, is_nifti_file, parse_session_bold_files
from bids.utils import listify
from pathlib import Path


logger = get_logger("nipype.workflow")

'''
Workflow illustration:

oceanfla_wf
|
|--space_MNI152_wf
| |
| \--|
|    |--run1_wf  \
|    |--run2_wf   -> regression_wf
|    ...         /
|    |--runN_wf /
|
|--space_fsLR_wf
| |
| \--|
|    |--run1_wf \
|    ...         -> regression_wf
|    |--runN_wf /
                         
One oceanfla workflow with func_space workflows as children, which in part have run-level
workflows as children that combine outputs to form a single regression workflow for each functional
space.
'''


def build_oceanfla_wf(subjects: list[str] | str | None, base_dir=Path | str):

    tasks = all_opts.task
    wf_name = f"oceanfla_tasks_{'-'.join(tasks)}_wf"
    fla_wf = Workflow(name=wf_name, base_dir=base_dir)

    subject_list = listify(subjects)
    if not subject_list:
        subject_list = collect_participants(
            bids_dir=all_opts.preproc_layout,
            participant_label=subject_list
        )

    start_node = Node(
        IdentityInterface(
            fields=["task"]
        ),
        name="task_start_node"
    )
    start_node.inputs.task = tasks

    for sub in subject_list:
        sessions = [None]
        if all_opts.session:
            sessions = [all_opts.session]
        else:
            bids_sessions = all_opts.preproc_layout.get_sessions(subject=sub)
            if len(bids_sessions) > 0:
                sessions = bids_sessions

        for ses in sessions:
            ses_wf = build_session_wf(subject=sub,
                                      session=ses)
            fla_wf.connect([
                (start_node, ses_wf, [
                    ("task", "inputnode.task")
                ])
            ])

    return fla_wf


def build_session_wf(subject, session=None):

    wf_name = f"sub_{subject}_{f'ses_{session}_' if session else ''}wf"
    logger.info(f"creating the session-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)

    input_node = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "task"
            ]
        ),
        name="inputnode"
    )
    input_node.inputs.subject = subject
    input_node.inputs.session = session

    space_run_info = parse_session_bold_files(layout=all_opts.preproc_layout,
                                              subject=subject,
                                              session=session,
                                              tasks=all_opts.task)
    space_dict = space_run_info[all_opts.func_space]
    func_space_wf = build_func_space_wf(func_space=all_opts.func_space,
                                        run_map=space_dict["runs"],
                                        file_extension=space_dict["extension"])

    workflow.connect([
        (input_node, func_space_wf, [
            ("subject", "inputnode.subject"),
            ("session", "inputnode.session"),
            ("task", "inputnode.task"),
        ])
    ])

    #### only doing one functional space at a time ######

    # for func_space, space_dict in space_run_info.items():
    #     func_space_wf = build_func_space_wf(func_space=func_space,
    #                                         run_map=space_dict["runs"],
    #                                         file_extension=space_dict["extension"])

    #     workflow.connect([
    #         (input_node, func_space_wf, [
    #             ("subject", "inputnode.subject"),
    #             ("session", "inputnode.session"),
    #             ("task", "inputnode.task"),
    #         ])
    #     ])

    ### TODO Reporting Stuff for this subject ###

    return workflow


def build_func_space_wf(func_space: str, run_map: dict, file_extension: str):

    # Define the workflow and the input node for this functional space
    wf_name = f"space_{func_space}_wf"
    logger.info(f"creating the functional-space-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)

    input_node = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "task"
            ]
        ),
        name="inputnode"
    )

    # Define the data grabber nodes to find the relevant files
    derivs_grabber = Node(
        BIDSDataGrabber(
            base_dir=all_opts.preproc_bids,
            datatype='func',
            # task=all_opts.task,
            output_query={
                'bold': {
                    'suffix': 'bold',
                    'space': func_space,
                    'extension': file_extension,
                },
                'confounds': {
                    'suffix': 'timeseries',
                    'desc': 'confounds',
                    'extension': '.tsv',
                }
            },
            load_layout=Path(
                all_opts.preproc_layout.connection_manager.database_file).parent
        ),
        name="derivs_bidssrc_node"
    )

    rawdata_grabber = Node(
        BIDSDataGrabber(
            base_dir=all_opts.raw_bids,
            datatype='func',
            # task=all_opts.task,
            output_query={
                'events': {
                    'suffix': 'events',
                    'extension': '.tsv'
                },
            },
            load_layout=Path(
                all_opts.raw_layout.connection_manager.database_file).parent
        ),
        name="raw_bidssrc_node"
    )

    surf_grabber = Node(
        BIDSDataGrabber(
            base_dir=all_opts.preproc_bids,
            datatype='anat',
            raise_on_empty=True,
            output_query={
                'lh': {
                    'suffix': 'midthickness',
                    'space': ["fsLR", "dhcpAsym"],
                    'extension': '.surf.gii',
                    'hemi': 'L'
                },
                'rh': {
                    'suffix': 'midthickness',
                    'space': ["fsLR", "dhcpAsym"],
                    'extension': '.surf.gii',
                    'hemi': 'R'
                }
            },
            load_layout=Path(
                all_opts.preproc_layout.connection_manager.database_file).parent
        ),
        name="surf_grabber_node"
    )

    lh_list_select_node = Node(
        Select(
            index=0
        ),
        name="lh_select_first_node"
    )
    rh_list_select_node = Node(
        Select(
            index=0
        ),
        name="rh_select_first_node"
    )

    # Connect the inputs to the data-grabber nodes
    workflow.connect([
        (input_node, derivs_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ]),
        (input_node, rawdata_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ])
    ])

    # if smoothing is requested
    if is_cifti_file(file_extension) and all_opts.fwhm:
        workflow.connect([
            (input_node, surf_grabber, [
                ("subject", "subject"),
                ("session", "session")
            ]),
            (surf_grabber, lh_list_select_node, [
                ("lh", "inlist")
            ]),
            (surf_grabber, rh_list_select_node, [
                ("rh", "inlist")
            ]),
        ])

    input_merging_node = Node(
        MergeUnique(),
        name="merge_run_data_node"
    )
    # Create a run-level workflow for each run that has this functional space
    input_num = 1
    for task, run_list in run_map.items():
        for run in run_list:
            run_level_wf = build_run_workflow(run=run,
                                              task=task,
                                              file_extension=file_extension)

            # Define a node to extract the run-specific files from the data-grabbers
            extract_task_run_group_node = Node(
                ExtractDataGroup(
                    task=task,
                    run=run
                ),
                name=f"extract_task_{task}_run_{run}_group_node"
            )

            # Connect the files to the run-level workflow
            workflow.connect([
                (input_node, run_level_wf, [
                    ("subject", "inputnode.subject"),
                    ("session", "inputnode.session"),
                ]),
                (derivs_grabber, extract_task_run_group_node, [
                    ("bold", "bold_list"),
                    ("confounds", "confounds_list")
                ]),
                (rawdata_grabber, extract_task_run_group_node, [
                    ("events", "events_list"),
                ]),
                (extract_task_run_group_node, run_level_wf, [
                    ("bold_file", "inputnode.bold_file"),
                    ("confounds_file", "inputnode.confounds_file"),
                    ("events_file", "inputnode.events_file"),
                ])
            ])

            if is_cifti_file(file_extension) and all_opts.fwhm:
                workflow.connect([
                    (lh_list_select_node, run_level_wf, [
                        ("out", "inputnode.lh_surf")
                    ]),
                    (rh_list_select_node, run_level_wf, [
                        ("out", "inputnode.rh_surf")
                    ])
                ])

            # Connect the output of the run-level workflow to the merging node
            for out_key in run_level_wf.get_node("outputnode").outputs.get().keys():
                workflow.connect(run_level_wf, f"outputnode.{out_key}",
                                 input_merging_node, f"{out_key}_x{input_num}")
            input_num += 1

    ## DO STUFF AFTER THE RUN-LEVEL WORKFLOWS ###
    # * concat run-level info
    # * run session-level glm
    regression_wf = build_regression_workflow(tasks=all_opts.task)

    workflow.connect([
        (input_merging_node, regression_wf, [
            ("bold_file", "inputnode.bold_files"),
            ("tmask_file", "inputnode.tmask_files"),
            ("design_matrix", "inputnode.event_matrices"),
            ("nuisance_matrix", "inputnode.nuisance_matrices"),
            ("include_in_regression", "inputnode.inclusion_list")
        ])
    ])

    ### Datasink for user outputs ###
    need_compress = file_extension.endswith(".gz")
    task_name = "-".join(run_map.keys())
    beta_weights_ds = Node(
        FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="effect",
            task=task_name,
            allowed_entities=("condition", "stat")
        ),
        name=f"{func_space}_beta_weights_ds"
    )
    workflow.connect([
        (regression_wf, beta_weights_ds, [
            ("outputnode.beta_files", "in_file"),
            ("outputnode.beta_labels", "condition")
        ]),
        (derivs_grabber, beta_weights_ds, [
            ("bold", "source_file")
        ])
    ])

    design_matrix_ds = Node(
        FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run"],
            desc="final",
            suffix="design",
            task=task_name,
        ),
        name=f"{func_space}_design_matrix_ds"
    )
    workflow.connect([
        (regression_wf, design_matrix_ds, [
            ("outputnode.design_matrix", "in_file"),
        ]),
        (derivs_grabber, design_matrix_ds, [
            ("confounds", "source_file")
        ])
    ])

    residual_bold_ds = Node(
        FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            extra_bids_patterns=all_opts.bids_patterns,
            compress=need_compress,
            dismiss_entities=["run", "den"],
            desc="glmResidual",
            task=task_name,
        ),
        name=f"{func_space}_residual_bold_ds"
    )
    workflow.connect([
        (regression_wf, residual_bold_ds, [
            ("outputnode.bold_file", "in_file"),
        ]),
        (derivs_grabber, residual_bold_ds, [
            ("bold", "source_file")
        ])
    ])

    return workflow


def build_run_workflow(run, task: str, file_extension: str):
    from oceanfla.interfaces.nuisance import make_regressor_run_specific

    ### Define the workflow and the inputnode ###
    wf_name = f"task_{task}_run_{run}_processsing_wf"
    logger.info(f"creating the run-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)
    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "bold_file",
                "confounds_file",
                "events_file",
                "lh_surf",
                "rh_surf"
            ]
        ),
        name="inputnode"
    )
    outputnode = Node(
        IdentityInterface(
            fields=[
                "bold_file",
                "design_matrix",
                "nuisance_matrix",
                "tmask_file",
                "include_in_regression"
            ]
        ),
        name="outputnode"
    )
    compress_files = file_extension.endswith(".gz")

    ### Create run-level temporal mask ###
    tmask_node = Node(
        MakeTmask(
            fd_threshold=all_opts.fd_threshold,
            minimum_unmasked_neighbors=all_opts.minimum_unmasked_neighbors,
            start_censoring=all_opts.start_censoring
        ),
        name="make_tmask_node"
    )

    ### Check that this run passes exclusion criteria ###
    exclusion_wf = build_exclusion_wf(run, task)

    workflow.connect([
        (inputnode, tmask_node, [
            ("confounds_file", "confounds_file")
        ]),
        (inputnode, exclusion_wf, [
            ("bold_file", "inputnode.bold_file"),
        ]),
        (tmask_node, exclusion_wf, [
            ("tmask_file", "inputnode.tmask_file")
        ]),
        (exclusion_wf, outputnode, [
            ("outputnode.include", "include_in_regression")
        ]),
        (tmask_node, outputnode, [
            ("tmask_file", "tmask_file"),
        ])
    ])

    ### Create run-level event matrix ###
    get_metadata_node = Node(
        ReadMetadataFile(
            fields=["RepetitionTime"],
            error_on_missing=True,
        ),
        name="get_metadata_node"
    )

    get_volumes_node = Node(
        GetVolumeCount,
        name="get_run_volumes_node"
    )
    get_volumes_node.inputs.brain_mask = all_opts.brain_mask

    events_matrix_node = Node(
        EventsMatrix(
            fir=all_opts.fir,
            hrf=all_opts.hrf,
            fir_vars=all_opts.fir_vars,
            hrf_vars=all_opts.hrf_vars,
            unmodeled=all_opts.unmodeled,
            parameters=all_opts.parametric_modulators
        ),
        name="events_matrix_node"
    )

    if all_opts.group or all_opts.ignore:
        modify_events_file_node = Node(
            ModifyEventsFile(
                trial_type_map=all_opts.group,
                removal_list=all_opts.ignore,
            ),
            name="modify_events_file_node"
        )
        workflow.connect([
            (inputnode, modify_events_file_node, [
                ("events_file", "events_file")
            ]),
            (modify_events_file_node, events_matrix_node, [
                ("events_out", "event_file")
            ])
        ])
    else:
        workflow.connect([
            (inputnode, events_matrix_node, [
                ("events_file", "event_file")
            ])
        ])

    if all_opts.repetition_time:
        events_matrix_node.inputs.tr = all_opts.repetition_time
    else:
        workflow.connect([
            (inputnode, get_metadata_node, [
                ("bold_file", "bids_file")
            ]),
            (get_metadata_node, events_matrix_node, [
                ("RepetitionTime", "tr")
            ])
        ])

    workflow.connect([
        (inputnode, get_volumes_node, [
            ("bold_file", "bold_in")
        ]),
        (get_volumes_node, events_matrix_node, [
            ("volumes", "volumes")
        ])
    ])

    ### Create run-level nuisance matrix ###
    nuisance_mat_node = Node(
        GenerateNuisanceMatrix(
            confounds_columns=all_opts.confounds,
            demean=(not all_opts.exclude_run_mean),
            linear_trend=(not all_opts.exclude_run_trend),
            spike_threshold=all_opts.fd_threshold if all_opts.spike_regression else None,
            volterra_lag=all_opts.volterra_lag,
            volterra_columns=all_opts.volterra_columns,
        ),
        name="nuisance_matrix_node"
    )
    workflow.connect([
        (inputnode, nuisance_mat_node, [
            ("confounds_file", "confounds_file")
        ])
    ])

    last_func_node = inputnode

    if all_opts.debug:
        # save out the working files
        event_matrix_ds = Node(FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            extra_bids_patterns=all_opts.bids_patterns,
            desc="modeled",
            suffix="events"
        ),
            name="event_matrix_ds"
        )
        workflow.connect([
            (inputnode, event_matrix_ds, [
                ("events_file", "source_file")
            ]),
            (events_matrix_node, event_matrix_ds, [
                ("events_matrix", "in_file")
            ])
        ])

        nuisance_matrix_ds = Node(FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            extra_bids_patterns=all_opts.bids_patterns,
            desc="nuisance",
        ),
            name="nuisance_matrix_ds"
        )
        workflow.connect([
            (inputnode, nuisance_matrix_ds, [
                ("confounds_file", "source_file")
            ]),
            (nuisance_mat_node, nuisance_matrix_ds, [
                ("nuisance_matrix", "in_file")
            ])
        ])

        make_tmask_tsv_node = Node(
            MakeTmaskTsv,
            name="make_tmask_tsv_node"
        )
        make_tmask_tsv_node.inputs.fd_threshold = all_opts.fd_threshold
        tmask_ds = Node(FLADataSink(
            base_directory=all_opts.derivs_dir,
            out_path_base=all_opts.derivs_subfolder,
            extra_bids_patterns=all_opts.bids_patterns,
            suffix="tmask"
        ),
            name="tmask_ds"
        )
        workflow.connect([
            (tmask_node, make_tmask_tsv_node, [
                ("tmask_file", "tmask_file")
            ]),
            (inputnode, tmask_ds, [
                ("events_file", "source_file")
            ]),
            (make_tmask_tsv_node, tmask_ds, [
                ("tmask_tsv", "in_file")
            ])
        ])

    ### Smooth the data if requested ###
    if all_opts.fwhm:
        smoothing_wf = build_smoothing_wf(
            run=run, task=task, file_extension=file_extension)

        workflow.connect([
            (inputnode, smoothing_wf, [
                ("subject", "inputnode.subject"),
                ("session", "inputnode.session"),
                ("lh_surf", "inputnode.lh_surf"),
                ("rh_surf", "inputnode.rh_surf"),
            ]),
            (last_func_node, smoothing_wf, [
                ("bold_file", "inputnode.bold_file")
            ]),
        ])
        last_func_node = smoothing_wf.get_node("outputnode")

        if all_opts.debug:
            smoothed_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="smooth",
                allowed_entities=("fwhm"),
                fwhm=str(all_opts.fwhm).replace(".", "p")
            ),
                name="smoothed_bold_ds"
            )
            workflow.connect([
                (smoothing_wf, smoothed_ds, [
                    ("outputnode.bold_file", "in_file"),
                ]),
                (inputnode, smoothed_ds, [
                    ("bold_file", "source_file")
                ])
            ])

    ### Percent signal change ###
    if all_opts.percent_change:
        # make psc node
        percent_change_node = Node(
            PercentChange(
                brain_mask=all_opts.brain_mask,
            ),
            name="percent_change_node"
        )

        workflow.connect([
            (last_func_node, percent_change_node, [
                ("bold_file", "bold_in")
            ]),
            (tmask_node, percent_change_node, [
                ("tmask_file", "tmask_in")
            ])
        ])
        last_func_node = percent_change_node
        if all_opts.debug:
            percent_change_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="percentChange",
            ),
                name="percent_change_bold_ds"
            )
            workflow.connect([
                (percent_change_node, percent_change_ds, [
                    ("bold_file", "in_file")
                ]),
                (inputnode, percent_change_ds, [
                    ("bold_file", "source_file")
                ])
            ])

    ### Nuisance regression ###
    if all_opts.nuisance_regression:
        regression_columns = [rc if rc not in all_opts.generic_nuisance_columns
                              else make_regressor_run_specific(rc, run=run, task=task)
                              for rc in all_opts.nuisance_regression]

        regression_wf = build_regression_workflow(
            tasks=task, run=run, regression_columns=regression_columns)

        workflow.connect([
            (last_func_node, regression_wf, [
                ("bold_file", "inputnode.bold_files")
            ]),
            (events_matrix_node, regression_wf, [
                ("events_matrix", "inputnode.event_matrices")
            ]),
            (tmask_node, regression_wf, [
                ("tmask_file", "inputnode.tmask_files")
            ]),
            (nuisance_mat_node, regression_wf, [
                ("nuisance_matrix", "inputnode.nuisance_matrices")
            ]),
            (regression_wf, outputnode, [
                ("outputnode.residual_design_matrix", "design_matrix"),
            ])
        ])
        outputnode.inputs.nuisance_matrix = None
        last_func_node = regression_wf.get_node("outputnode")

        if all_opts.debug:
            regressed_bold_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="nuisanceRegressed"
            ),
                name="regressed_bold_ds"
            )
            workflow.connect([
                (regression_wf, regressed_bold_ds, [
                    ("outputnode.bold_file", "in_file")
                ]),
                (inputnode, regressed_bold_ds, [
                    ("bold_file", "source_file")
                ])
            ])

            nuisance_betas_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="nuisanceRegression",
                suffix="boldmap",
                stat="effect",
                allowed_entities=("condition", "stat", "den")
            ),
                name="nuisance_betas_ds"
            )
            workflow.connect([
                (regression_wf, nuisance_betas_ds, [
                    ("outputnode.beta_files", "in_file"),
                    ("outputnode.beta_labels", "condition")
                ]),
                (inputnode, nuisance_betas_ds, [
                    ("bold_file", "source_file")
                ])
            ])

            design_file_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                desc="nuisanceRegression",
                suffix="design"
            ),
                name="design_file_ds"
            )
            workflow.connect([
                (regression_wf, design_file_ds, [
                    ("outputnode.design_matrix", "in_file"),
                ]),
                (inputnode, design_file_ds, [
                    ("confounds_file", "source_file")
                ])
            ])

    else:
        workflow.connect([
            (events_matrix_node, outputnode, [
                ("events_matrix", "design_matrix")
            ]),
            (nuisance_mat_node, outputnode, [
                ("nuisance_matrix", "nuisance_matrix")
            ])
        ])

    ### Bandpass filter ###
    if all_opts.highpass or all_opts.lowpass:

        filter_node = Node(
            FilterData(
                high_pass=all_opts.highpass,
                low_pass=all_opts.lowpass,
                padtype=all_opts.filter_padtype,
                padlen=all_opts.filter_padlen,
                brain_mask=all_opts.brain_mask
            ),
            name="filtering_node"
        )

        workflow.connect([
            (last_func_node, filter_node, [
                ("bold_file", "bold_in")
            ]),
            (tmask_node, filter_node, [
                ("tmask_file", "tmask_in")
            ])
        ])

        if all_opts.repetition_time:
            filter_node.inputs.tr = all_opts.repetition_time
        else:
            workflow.connect([
                (get_metadata_node, filter_node, [
                    ("RepetitionTime", "tr")
                ])
            ])

        if all_opts.debug:
            filter_ds = Node(FLADataSink(
                base_directory=all_opts.derivs_dir,
                out_path_base=all_opts.derivs_subfolder,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                allowed_entites=("hp", "lp"),
                dismiss_entities=["den"],
                desc="filtered",
            ),
                name="filtered_bold_ds"
            )
            if all_opts.highpass:
                filter_ds.inputs.hp = str(all_opts.highpass).replace(".", "p")
            if all_opts.lowpass:
                filter_ds.inputs.lp = str(all_opts.lowpass).replace(".", "p")

            workflow.connect([
                (filter_node, filter_ds, [
                    ("bold_file", "in_file")
                ]),
                (inputnode, filter_ds, [
                    ("bold_file", "source_file")
                ])
            ])

        last_func_node = filter_node

    ### Connect final bold output ###
    workflow.connect([
        (last_func_node, outputnode, [
            ("bold_file", "bold_file")
        ])
    ])

    return workflow


def build_regression_workflow(tasks, run=None, regression_columns=None):

    tasks = listify(tasks)
    wf_label = f"task_{'-'.join(tasks)}"
    if run:
        wf_label += f"_run_{run}"
    workflow = Workflow(name=f"{wf_label}_regression_wf")

    inputnode = Node(
        IdentityInterface(
            fields=[
                "bold_files",
                "event_matrices",
                "tmask_files",
                "nuisance_matrices",
                "regressor_columns",
                "inclusion_list"
            ]
        ),
        name="inputnode"
    )
    inputnode.inputs.regressor_columns = regression_columns

    outputnode = Node(
        IdentityInterface(
            fields=[
                "beta_files",
                "beta_labels",
                "bold_file",
                "design_matrix",
                "residual_design_matrix"
            ]
        ),
        name="outputnode"
    )
    need_mean = (not all_opts.no_global_mean) if run is None else all_opts.exclude_run_mean

    concat_data_node = Node(
        ConcatRegressionData(
            include_global_mean=need_mean,
            tasks=tasks,
            brain_mask=all_opts.brain_mask,
        ),
        name="concat_data_node"
    )

    stdscale = (all_opts.stdscale_glm in ["both", "runlevel"]
                ) if run else (
        all_opts.stdscale_glm in ["both", "seslevel"])
    glm_node = Node(
        RunGLMRegression(
            stdscale=stdscale,
            brain_mask=all_opts.brain_mask
        ),
        name="glm_regression_node"
    )

    workflow.connect([
        (inputnode, concat_data_node, [
            ("bold_files", "bold_files_in"),
            ("event_matrices", "event_matrices"),
            ("nuisance_matrices", "nuisance_matrices"),
            ("regressor_columns", "regressor_columns"),
            ("inclusion_list", "inclusion_list")
        ]),
        (concat_data_node, glm_node, [
            ("bold_file", "bold_file_in"),
            ("design_matrix", "design_matrix"),
            ("tmask_file", "tmask_file")
        ]),
        (concat_data_node, outputnode, [
            ("design_matrix", "design_matrix"),
            ("residual_design_matrix", "residual_design_matrix"),
        ]),
        (glm_node, outputnode, [
            ("beta_files", "beta_files"),
            ("beta_labels", "beta_labels"),
            ("residual_bold_file", "bold_file")
        ])
    ])

    if all_opts.fd_censoring:
        workflow.connect(inputnode, "tmask_files",
                         concat_data_node, "tmask_files_in")
    else:
        concat_data_node.inputs.tmask_files_in = None

    return workflow


def build_exclusion_wf(run, task):

    workflow = Workflow(name=f"task_{task}_run_{run}_exclusion_wf")
    inputnode = Node(
        IdentityInterface(
            fields=[
                "bold_file",
                "tmask_file"
            ]
        ),
        name="inputnode"
    )
    outputnode = Node(
        IdentityInterface(
            fields=[
                "include"
            ]
        ),
        name="outputnode"
    )

    ### Create tSNR node ###
    tsnr_check_node = Node(
        CheckRuntSNR(
            tsnr_threshold=all_opts.min_average_tsnr,
            brain_mask=all_opts.brain_mask,
        ),
        name="check_tsnr_node"
    )

    frame_retention_check_node = Node(
        CheckRunRetention(
            retention_threshold=all_opts.run_exclusion_threshold,
            start_censoring=all_opts.start_censoring
        ),
        name="check_frame_retention_node"
    )

    validation_merging_node = Node(
        MergeUnique(),
        name="merge_validations_node"
    )

    def and_all_func(validation_list): return all(validation_list)
    check_validation_node = Node(
        Function(
            function=and_all_func,
            input_names=["validation_list"],
            output_names="include"
        ),
        name="check_validation_node"
    )

    workflow.connect([
        (inputnode, tsnr_check_node, [
            ("bold_file", "bold_file"),
            ("tmask_file", "tmask_file")
        ]),
        (inputnode, frame_retention_check_node, [
            ("tmask_file", "tmask_file")
        ]),
        (tsnr_check_node, validation_merging_node, [
            ("valid", "valid_x1")
        ]),
        (frame_retention_check_node, validation_merging_node, [
            ("valid", "valid_x2")
        ]),
        (validation_merging_node, check_validation_node, [
            ("valid", "validation_list")
        ]),
        (check_validation_node, outputnode, [
            ("include", "include")
        ])
    ])

    return workflow


def build_smoothing_wf(run, task: str, file_extension: str):
    from nibabel.processing import fwhm2sigma

    ### Define the workflow and the inputnode ###
    geom_label = "surface" if is_cifti_file(file_extension) else "volume"
    workflow = Workflow(
        name=f"task_{task}_run_{run}_{geom_label}_smoothing_wf")
    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "bold_file",
                "lh_surf",
                "rh_surf"
            ]
        ),
        name="inputnode"
    )
    outputnode = Node(
        IdentityInterface(
            fields=[
                "bold_file",
            ]
        ),
        name="outputnode"
    )

    if is_cifti_file(file_extension):
        surface_smooth_node = Node(
            CiftiSmooth(
                sigma_surf=fwhm2sigma(all_opts.fwhm),
                sigma_vol=fwhm2sigma(all_opts.fwhm),
                direction="COLUMN",
            ),
            name="surface_smooth_node"
        )
        surface_smooth_node.inputs.environ.update(
            {"OMP_NUM_THREADS": str(all_opts.n_procs)})

        workflow.connect([
            (inputnode, surface_smooth_node, [
                ("bold_file", "in_file")
            ]),
            (inputnode, surface_smooth_node, [
                ("lh_surf", "left_surf")
            ]),
            (inputnode, surface_smooth_node, [
                ("rh_surf", "right_surf")
            ]),
            (surface_smooth_node, outputnode, [
                ("out_file", "bold_file")
            ])
        ])

    elif is_nifti_file(file_extension):
        volume_smooth_node = Node(
            VolumeSmooth(
                kernel=fwhm2sigma(all_opts.fwhm),
                fwhm=True,
            ),
            name="volume_smooth_node"
        )
        volume_smooth_node.inputs.environ.update(
            {"OMP_NUM_THREADS": str(all_opts.n_procs)})
        workflow.connect([
            (inputnode, volume_smooth_node, [
                ("bold_file", "volume_in")
            ]),
            (volume_smooth_node, outputnode, [
                ("volume_out", "bold_file")
            ])
        ])

    return workflow
