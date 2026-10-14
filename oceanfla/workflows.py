from nipype import Node, Workflow, Function
from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.utility import IdentityInterface, Select
# from niworkflows.utils.bids import collect_participants
from oceanfla.interfaces.reporting import PlotDesign, ReportExclusions
from oceanfla.interfaces.utility import BIDSSource, FLADataSink
from oceanfla.interfaces.clean import FilterData, PercentChange
from oceanfla.interfaces.events import EventsMatrix, FindContinuousVars, ModifyEventsFile
from oceanfla.interfaces.exclusions import CheckExclusionFile, CheckRunRetention, CheckRuntSNR, GetIncludedFiles, MakeRunExclusionTable
from oceanfla.interfaces.nuisance import GenerateNuisanceMatrix
from oceanfla.interfaces.regression import CombineFIRBetas, ConcatRegressionData, MakeRunDesign, RunGLMRegression
from oceanfla.interfaces.tmask import FindDscans, MakeTmask, RemoveMotionMasking, make_tmask_tsv
from oceanfla.interfaces.utility import MergeUnique, ExtractDataGroup
from oceanfla.config import all_opts, get_logger
from oceanfla.interfaces.workbench_utils import VolumeSmooth, SurfaceSmooth
from oceanfla.utilities import is_cifti_file, is_nifti_file, parse_session_bold_files
from bids.utils import listify
from pathlib import Path

logger = get_logger("nipype.workflow")


# Workflow illustration:

# oceanfla_wf
# | |
# | \--|
# |    |--taskX_run1_design_wf
# |    ...
# |    |-taskX_runN_design_wf   \
# |                             |
# |--space_MNI152_wf            |
# | |                           |
# | \--|                        |
# |    |--run1_wf  \            v
# |    |--run2_wf   -> regression_wf
# |    ...         /
# |    |--runN_wf /
# |
# |--space_fsLR_wf
# | |
# | \--|
# |    |--run1_wf \
# |    ...         -> regression_wf
# |    |--runN_wf /
                         
# One oceanfla workflow with func_space workflows as children, which in part have run-level
# workflows as children that combine outputs to form a single regression workflow for each functional
# space.



def build_oceanfla_wf(subjects: list[str] | str | None, base_dir:Path | str):

    tasks = all_opts.task
    wf_name = f"oceanfla_task_{all_opts.task_rename}_wf"
    fla_wf = Workflow(name=wf_name, base_dir=base_dir)

    subject_list = listify(subjects)
    if not subject_list:
        subject_list = sorted([d.name.split("sub-")[-1] for d in Path(all_opts.preproc_layout.root).glob("sub-*") if d.is_dir()])

    start_node = Node(
        IdentityInterface(
            fields=["task"]
        ),
        name="task_start_node"
    )
    start_node.inputs.task = tasks

    nothing_to_do = True
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
            if ses_wf:
                fla_wf.connect([
                    (start_node, ses_wf, [
                        ("task", "inputnode.task")
                    ])
                ])
                nothing_to_do = False
    
    if nothing_to_do:
        logger.warning("NO WORKFLOWS WERE CREATED, EXITING NOW")
        return None

    return fla_wf


def build_session_wf(subject, session=None):

    wf_name = f"sub_{subject}_{f'ses_{session}_' if session else ''}wf"
    logger.info(f"creating the session-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)

    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "task"
            ]
        ),
        name="inputnode"
    )
    inputnode.inputs.subject = subject
    inputnode.inputs.session = session

    space_run_info, ref_space = parse_session_bold_files(
        layout=all_opts.preproc_layout,
        subject=subject,
        session=session,
        tasks=all_opts.task,
        tr=all_opts.repetition_time)
    
    if all_opts.func_space not in space_run_info:
        logger.warning(f"NO BOLD RUNS FOUND FOR SUBJECT:{subject}, SESSION:{session}, TEMPLATE_SPACE:{all_opts.func_space}")
        return None

    if all_opts.func_space:
        ref_space = all_opts.func_space
    
    confounds_grabber = Node(
        BIDSDataGrabber(
            base_dir=all_opts.preproc_bids,
            datatype='func',
            output_query={
                'confounds': {
                    'suffix': 'timeseries',
                    'desc': 'confounds',
                    'extension': '.tsv',
                }
            },
            load_layout=Path(
                all_opts.preproc_layout.connection_manager.database_file).parent
        ),
        name="confounds_bidssrc_node"
    )

    events_grabber = Node(
        BIDSDataGrabber(
            base_dir=all_opts.raw_bids,
            datatype='func',
            output_query={
                'events': {
                    'suffix': 'events',
                    'extension': '.tsv'
                },
            },
            load_layout=Path(
                all_opts.raw_layout.connection_manager.database_file).parent
        ),
        name="events_bidssrc_node"
    )

    # Connect the inputs to the data-grabber nodes
    workflow.connect([
        (inputnode, confounds_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ]),
        (inputnode, events_grabber, [
            ("subject", "subject"),
            ("session", "session"),
            ("task", "task")
        ])
    ])
    
    design_merging_node = Node(
        MergeUnique(),
        name="merge_design_run_data"
    )
    # for task, bold_list in space_run_info[list(space_run_info.keys())[0]].items():
    merge_index = 0
    for task, bold_list in space_run_info[ref_space].items():
        if task == "extension":
            continue
        for bold_info in bold_list:
            # bold_bids = get_bids_file(bold_run)
            run = bold_info["run"]

            bold_run_identity_node = Node(
                IdentityInterface(
                    fields=[
                        "tr",
                        "volumes"
                    ]
                ),
                name=f"{task}_run_{run}_bold_identity_node"
            )
            bold_run_identity_node.inputs.tr = bold_info["tr"]
            bold_run_identity_node.inputs.volumes = bold_info["volumes"]

            ses_design_wf = build_ses_design_wf(run, task)

            extract_task_run_souce_node = Node(
                ExtractDataGroup(
                    task=task,
                    run=run
                ),
                name=f"extract_task_{task}_run_{run}_source_files_node"
            )

            workflow.connect([
                (inputnode, ses_design_wf, [
                    ("subject", "inputnode.subject"),
                    ("session", "inputnode.session"),
                ]),
                (bold_run_identity_node, ses_design_wf, [
                    ("tr", "inputnode.tr"),
                    ("volumes", "inputnode.volumes")
                ]),
                (confounds_grabber, extract_task_run_souce_node, [
                    ("confounds", "confounds")
                ]),
                (events_grabber, extract_task_run_souce_node, [
                    ("events", "events"),
                ]),
                (extract_task_run_souce_node, ses_design_wf, [
                    ("confounds", "inputnode.confounds_file"),
                    ("events", "inputnode.events_file"),
                ]),
                (extract_task_run_souce_node, design_merging_node, [
                    ("confounds", f"confounds_x{merge_index}")
                ])
            ])

            # Connect the output of the ses-design workflow to the merging node
            for out_key in ses_design_wf.get_node("outputnode").outputs.get().keys():
                workflow.connect(ses_design_wf, f"outputnode.{out_key}",
                                 design_merging_node, f"{out_key}_x{merge_index}")

            merge_index += 1


    #### only doing one functional space at a time ######
    # for func_space, space_dict in space_run_info.items():
    space_dict = space_run_info[all_opts.func_space]
    file_extension = space_dict.pop("extension")
    func_space_wf = build_func_space_wf(func_space=all_opts.func_space,
                                        run_map=space_dict,
                                        file_extension=file_extension)

    workflow.connect([
        (inputnode, func_space_wf, [
            ("subject", "inputnode.subject"),
            ("session", "inputnode.session"),
        ]), 
        (design_merging_node, func_space_wf, [
            ("main_design", "inputnode.main_design_files"),
            ("nuisance_design", "inputnode.nuisance_design_files"),
            ("tmask_file", "inputnode.tmask_files"),
            ("confounds", "inputnode.confounds_files")
        ])
    ])

    ### TODO Reporting Stuff for this subject ###

    return workflow


def build_ses_design_wf(run, task):
    from oceanfla.interfaces.nuisance import make_regressor_run_specific

    workflow = Workflow(name=f"task_{task}_run_{run}_design_wf")

    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "confounds_file",
                "events_file",
                "tr",
                "volumes"
            ]
        ),
        name="inputnode"
    )

    outputnode = Node(
        IdentityInterface(
            fields=[
                "main_design",
                "tmask_file",
                "nuisance_design"
            ]
        ),
        name="outputnode"
    )

    ### Create run-level temporal mask ###
    tmask_node = Node(
        MakeTmask(
            fd_threshold=all_opts.fd_threshold,
            minimum_unmasked_neighbors=all_opts.minimum_unmasked_neighbors,
            start_censoring=all_opts.start_censoring
        ),
        name="make_tmask_node"
    )
    workflow.connect([
        (inputnode, tmask_node, [
            ("confounds_file", "confounds_file")
        ]),
    ]) 


    ### save out the run-level tmask files
    make_tmask_tsv_node = Node(
        Function(
            function=make_tmask_tsv,
            input_names=["tmask_file", "fd_threshold", "execute"],
            output_names=["tmask_tsv"]
        ),
        name="make_tmask_tsv_node"
    )
    make_tmask_tsv_node.inputs.fd_threshold = all_opts.fd_threshold
    tmask_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            suffix="tmask",
            FramewiseDisplacementThreshold=f"{all_opts.fd_threshold}mm",
            NonSteadyStateVolumesDiscardedFromStart=all_opts.start_censoring,
            NeighborMaskingLength=all_opts.minimum_unmasked_neighbors,
            Description="Temporal Mask"
            
        ),  
        name="tmask_ds"
    )
    tmask_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name="tmask_sources_node"
    )
    workflow.connect([
        (inputnode, tmask_sources_node, [
            ("confounds_file", "file_1")
        ]),
        (tmask_sources_node, tmask_ds,[
            ("sources", "Sources")
        ]),
        (tmask_node, make_tmask_tsv_node, [
            ("tmask_file", "tmask_file")
        ]),
        (inputnode, tmask_ds, [
            ("events_file", "source_file")
        ]),
        (make_tmask_tsv_node, tmask_ds, [
            ("tmask_tsv", "in_file")
        ]),
        (tmask_ds, outputnode, [
            ("out_file", "tmask_file")
        ])
    ])
    
    ### Find Dummy scans if supplied ###
    if all_opts.dscans_path:
        find_dscans_node = Node(
            FindDscans(
                dscans_directory=all_opts.dscans_path,
            ),
            name="find_dscans_node"
        )
        workflow.connect([
            (inputnode, find_dscans_node, [
                ("events_file", "source_file")
            ]),
            (find_dscans_node, tmask_node, [
                ("dscans_file", "dscans_tsv")
            ]),
            (find_dscans_node, tmask_sources_node,[
                ("dscans_file", "file_2")
            ])
        ])

    ### Create run-level event matrix ###
    events_matrix_node = Node(
        EventsMatrix(
            fir=all_opts.fir,
            hrf=all_opts.hrf,
            fir_vars=all_opts.fir_vars,
            hrf_vars=all_opts.hrf_vars,
            unmodeled=all_opts.unmodeled,
            parameters=all_opts.parametric_modulators,
            continuous_columns=all_opts.continuous_columns
        ),
        name="events_matrix_node"
    )
    workflow.connect([
        (inputnode, events_matrix_node, [
            ("tr", "tr"),
            ("volumes", "volumes")
        ])
    ])

    ### Group task regressors if requested ###
    if all_opts.group:
        modify_events_file_node = Node(
            ModifyEventsFile(
                trial_type_map=all_opts.group,
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

    ### Get other continuous variables if requested ###
    if all_opts.continuous_vars:
        find_continuous_vars_node = Node(
            FindContinuousVars(
                continuous_var_path=all_opts.continuous_vars,
                task=task,
                run=run
            ),
            name="find_continuous_vars_node"
        )
        workflow.connect([
            (inputnode, find_continuous_vars_node, [
                ("subject", "subject"),
                ("session", "session")
            ]),
            (find_continuous_vars_node, events_matrix_node, [
                ("continuous_vars_file", "continuous_vars_file")
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
            volterra_columns=all_opts.volterra_columns
        ),
        name="nuisance_matrix_node"
    )
    workflow.connect([
        (inputnode, nuisance_mat_node, [
            ("confounds_file", "confounds_file")
        ])
    ])

    ### created the needed design files ###
    nuisance_regressors = None
    if all_opts.nuisance_regression:
        nuisance_regressors = [rc if rc not in all_opts.generic_nuisance_columns 
                              else make_regressor_run_specific(rc, run=run, task=task) 
                              for rc in all_opts.nuisance_regression]
        if ("mean" not in all_opts.nuisance_regression) and (not all_opts.exclude_run_mean):
            nuisance_regressors.append(make_regressor_run_specific("mean", run=run, task=task))

    make_run_designs_node = Node(
        MakeRunDesign(
            nuisance_regressors=nuisance_regressors,
            removal_list=all_opts.ignore,
            volterra_columns=all_opts.volterra_columns
        ),
        name="make_run_designs_node"
    )
    workflow.connect([
        (events_matrix_node, make_run_designs_node, [
            ("events_matrix", "event_matrix"),
        ]),
        (nuisance_mat_node, make_run_designs_node, [
            ("nuisance_matrix", "nuisance_matrix")
        ]),
    ])

    hrf_model_desc = f"{all_opts.hrf[0]:02.2f}s to Peak & {all_opts.hrf[1]:02.2f}s Undershoot Duration" if all_opts.hrf else "",
    fir_model_desc = f"{all_opts.fir:02d} Frames" if all_opts.fir else ""

    ### Save these other run-level files out if requested ###
    main_design_ds = Node(FLADataSink(
        base_directory=all_opts.datasink_path.parent,
        out_path_base=all_opts.datasink_path.name,
        extra_bids_patterns=all_opts.bids_patterns,
        desc="event",
        suffix="design",
        HRFModel=hrf_model_desc,
        FIRModel=fir_model_desc,
    ),
        name="main_design_ds"
    )
    design_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name="main_design_sources_node"
    )
    workflow.connect([
        (inputnode, design_sources_node, [
            ("events_file", "file_1"),
            ("confounds_file", "file_2")
        ]),
        (design_sources_node, main_design_ds, [
            ("sources", "Sources")
        ]),
        (inputnode, main_design_ds, [
            ("events_file", "source_file"),
            ("tr", "RepetitionTime")
        ]),
        (make_run_designs_node, main_design_ds, [
            ("main_design", "in_file")
        ]),
        (main_design_ds, outputnode, [
            ("out_file", "main_design")
        ])
    ])

    if nuisance_regressors:
        nuisance_design_ds = Node(FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            desc="nuisance",
            suffix="design",
            HRFModel=hrf_model_desc,
            FIRModel=fir_model_desc,
        ),
            name="nuisance_design_ds"
        )
        workflow.connect([
            (design_sources_node, nuisance_design_ds, [
                ("sources", "Sources")
            ]),
            (inputnode, nuisance_design_ds, [
                ("confounds_file", "source_file"),
                ("tr", "RepetitionTime")
            ]),
            (make_run_designs_node, nuisance_design_ds, [
                ("nuisance_design", "in_file")
            ]),
            (nuisance_design_ds, outputnode, [
                ("out_file", "nuisance_design")
            ])
        ])
    else:
        workflow.connect([
            (make_run_designs_node, outputnode, [
                ("nuisance_design", "nuisance_design")
            ])
        ])

    
    return workflow


def build_func_space_wf(func_space: str, run_map: dict, file_extension: str):

    # Define the workflow and the input node for this functional space
    wf_name = f"space_{func_space}_wf"
    logger.info(f"creating the functional-space-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)

    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "main_design_files",
                "tmask_files",
                "nuisance_design_files",
                "confounds_files"
            ]
        ),
        name="inputnode"
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

    # if smoothing is requested
    if is_cifti_file(file_extension) and all_opts.fwhm:
        workflow.connect([
            (inputnode, surf_grabber, [
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
    for task, bold_list in run_map.items():
        for bold_info in bold_list:
            # bold_bids = get_bids_file(bold_run)
            run = bold_info["run"]

            bold_run_identity_node = Node(
                IdentityInterface(
                    fields=[
                        "bold_file",
                        "tr"
                    ]
                ),
                name=f"{task}_run_{run}_bold_identity_node"
            )
            bold_run_identity_node.inputs.bold_file = bold_info["bold_file"]
            bold_run_identity_node.inputs.tr = bold_info["tr"]

            # Define a node to extract the run-specific files from the data-grabbers
            extract_task_run_design_node = Node(
                ExtractDataGroup(
                    task=task,
                    run=run
                ),
                name=f"extract_task_{task}_run_{run}_design_files_node"
            )
            workflow.connect([
                (inputnode, extract_task_run_design_node, [
                    ("main_design_files", "main_design"),
                    ("nuisance_design_files", "nuisance_design"),
                    ("tmask_files", "tmask_file"),
                    ("confounds_files", "confounds")
                ])
            ])

            run_exclusion_wf = build_exclusion_wf(run=run,
                                                  task=task)
            # Connect the needed files to the exclusion workflow
            workflow.connect([
                (inputnode, run_exclusion_wf, [
                    ("subject", "inputnode.subject"),
                    ("session", "inputnode.session")
                ]),
                (bold_run_identity_node, run_exclusion_wf, [
                    ("bold_file", "inputnode.bold_file")
                ]),
                (extract_task_run_design_node, run_exclusion_wf, [
                    ("tmask_file", "inputnode.tmask_file")
                ])
            ])


            run_level_wf = build_run_workflow(run=run,
                                              task=task,
                                              func_space=func_space,
                                              file_extension=file_extension)
            # Connect the files to the run-level workflow
            workflow.connect([
                (inputnode, run_level_wf, [
                    ("subject", "main_inputnode.subject"),
                    ("session", "main_inputnode.session"),
                ]),
                (extract_task_run_design_node, run_level_wf, [
                    ("tmask_file", "main_inputnode.tmask_file"),
                    ("nuisance_design", "main_inputnode.nuisance_design"),
                ]),
                (bold_run_identity_node, run_level_wf, [
                    ("bold_file", "main_inputnode.bold_file"),
                    ("tr", "main_inputnode.tr")
                ]),
                (run_exclusion_wf, run_level_wf, [
                    ("outputnode.include", f"main_inputnode.include")
                ])
            ])

            if is_cifti_file(file_extension) and all_opts.fwhm:
                workflow.connect([
                    (lh_list_select_node, run_level_wf, [
                        ("out", "main_inputnode.lh_surf")
                    ]),
                    (rh_list_select_node, run_level_wf, [
                        ("out", "main_inputnode.rh_surf")
                    ])
                ])

            # Connect the output of the run-level workflows to the merging node.
            # The merge must consume the workflow's final result node, not the nested
            # regression workflow output.
            workflow.connect([
                (bold_run_identity_node, input_merging_node, [
                    ("tr", f"tr_x{input_num}")
                ]),
                (run_level_wf, input_merging_node, [
                    ("final_outputnode.processed_bold", f"bold_file_x{input_num}")
                ]),
                (extract_task_run_design_node, input_merging_node, [
                    ("tmask_file", f"tmask_file_x{input_num}"),
                    ("main_design", f"design_matrix_x{input_num}"),
                    ("confounds", f"confounds_x{input_num}")
                ]),
                (run_exclusion_wf, input_merging_node, [
                    ("outputnode.include", f"include_x{input_num}"),
                    ("outputnode.exclusion_table", f"exclusion_table_x{input_num}")
                ])
            ])

            input_num += 1

    ## DO STUFF AFTER THE RUN-LEVEL WORKFLOWS ###
    # * concat run-level info
    # * run session-level glm
    regression_wf = build_regression_workflow(
        task=all_opts.task_rename,
        need_intercept=True
    )

    workflow.connect([
        (input_merging_node, regression_wf, [
            ("bold_file", "inputnode.bold_files"),
            ("tmask_file", "inputnode.tmask_files"),
            ("design_matrix", "inputnode.design_matrices"),
            ("include", "inputnode.inclusion_list")
        ])
    ])

    find_included_sources_node = Node(
        GetIncludedFiles(),
        name="find_included_sources_node"
    )
    workflow.connect([
        (input_merging_node, find_included_sources_node, [
            ("include", "included"),
            ("bold_file", "bold_files"),
            ("tmask_file", "tmask_files"),
            ("design_matrix", "design_matrices"),
            ("confounds", "confounds"),
        ])
    ])


    ### Datasink for user outputs ###
    bold_runs_list = [bi["bold_file"] for bold_list in run_map.values() for bi in bold_list]
    need_compress = file_extension.endswith(".gz")

    # save the session-level tmask
    tmask_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="tmask",
            task=all_opts.task_rename,
            source_file=bold_runs_list,
            Description="Concatenated Temporal Mask"
        ),
        name=f"{func_space}_tmask_ds"
    )
    tmask_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name=f"{func_space}_tmask_sources_node"
    )
    workflow.connect([
        (find_included_sources_node, tmask_sources_node, [
            ("tmask_files", "file_1")
        ]),
        (tmask_sources_node, tmask_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, tmask_ds, [
            ("outputnode.tmask_file", "in_file"),
            ("outputnode.execute", "execute")
        ])
    ])

    # save design and reporting info 
    design_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run", "den"],
            desc="final",
            space=func_space,
            suffix="design",
            task=all_opts.task_rename,
            source_file=bold_runs_list
        ),
        name=f"{func_space}_design_ds"
    )
    design_plot_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run", "den"],
            desc="final",
            space=func_space,
            suffix="design",
            task=all_opts.task_rename,
            source_file=bold_runs_list
        ),
        name=f"{func_space}_design_plot_ds"
    )
    design_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name=f"{func_space}_design_sources_node"
    )
    design_corr_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run", "den"],
            desc="final",
            space=func_space,
            suffix="correlations",
            task=all_opts.task_rename,
            source_file=bold_runs_list
        ),
        name=f"{func_space}_design_corr_ds"
    )
    inclusion_report_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run", "den"],
            desc="inclusion",
            space=func_space,
            suffix="report",
            task=all_opts.task_rename,
            source_file=bold_runs_list
        ),
        name=f"{func_space}_inclusion_report_ds"
    )
    unmasked_design_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["run", "den"],
            desc="unmasked",
            space=func_space,
            suffix="design",
            task=all_opts.task_rename,
            source_file=bold_runs_list
        ),
        name=f"{func_space}_unmasked_design_ds"
    )
    unmasked_design_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name=f"{func_space}_unmasked_design_sources_node"
    )

    reporting_wf = build_reporting_workflow(task=all_opts.task_rename)
    workflow.connect([
        (regression_wf, reporting_wf, [
            ("outputnode.design_matrix", "inputnode.design_matrix"),
            ("outputnode.tmask_file", "inputnode.final_tmask_file"),
            ("outputnode.execute", "inputnode.execute")
        ]),
        (input_merging_node, reporting_wf, [
            ("tmask_file", "inputnode.run_tmask_files"),
            ("confounds", "inputnode.confounds_files"),
            ("include", "inputnode.inclusion_list"),
            ("exclusion_table", "inputnode.exclusion_tables")
        ]),
        (find_included_sources_node, design_sources_node, [
            ("design_matrices", "file_1")
        ]),
        (tmask_ds, design_sources_node, [
            ("out_file", "file_2")
        ]),
        (design_sources_node, design_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, design_ds, [
            ("outputnode.design_matrix", "in_file"),
            ("outputnode.execute", "execute"),
        ]),
        (find_included_sources_node, unmasked_design_sources_node, [
            ("design_matrices", "file_1")
        ]),
        (unmasked_design_sources_node, unmasked_design_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, unmasked_design_ds, [
            ("outputnode.unmasked_design_matrix", "in_file"),
            ("outputnode.execute", "execute")
        ]),
        (reporting_wf, design_plot_ds, [
            ("outputnode.design_plot", "in_file")
        ]),
        (regression_wf, design_plot_ds, [
            ("outputnode.execute", "execute"),
        ]),
        (reporting_wf, design_corr_ds, [
            ("outputnode.design_correlations", "in_file"),
        ]),
        (regression_wf, design_corr_ds, [
            ("outputnode.execute", "execute"),
        ]),
        (reporting_wf, inclusion_report_ds, [
            ("outputnode.exclusion_report", "in_file")
        ])
    ])

    # save the beta weights and test results
    beta_sources_node = Node(
        BIDSSource(
            raw_dataset=all_opts.raw_bids,
            preprocessed_dataset=all_opts.preproc_bids,
            datasink_path=all_opts.datasink_path,
            work_path=all_opts.work
        ),
        name = "beta_sources_node"
    )

    beta_weights_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="effect",
            task=all_opts.task_rename,
            allowed_entities=("condition", "stat"),
            source_file=bold_runs_list,
            Test="OLS"
        ),
        name=f"{func_space}_beta_weights_ds"
    )
    workflow.connect([
        (find_included_sources_node, beta_sources_node, [
            ("bold_files", "file_1")
        ]),
        (design_ds, beta_sources_node, [
            ("out_file", "file_2")
        ]),
        (tmask_ds, beta_sources_node, [
            ("out_file", "file_3")
        ]),
        (beta_sources_node, beta_weights_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, beta_weights_ds, [
            ("outputnode.execute", "execute")
        ])
    ])

    t_stat_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="t",
            task=all_opts.task_rename,
            allowed_entities=("condition", "stat"),
            source_file=bold_runs_list,
            Test="Wald T-test"
        ),
        name=f"{func_space}_t_stat_ds"
    )
    workflow.connect([
        (beta_sources_node, t_stat_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, t_stat_ds, [
            ("outputnode.execute", "execute")
        ])
    ])

    p_val_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="p",
            task=all_opts.task_rename,
            allowed_entities=("condition", "stat"),
            source_file=bold_runs_list,
            Type="2-tailed",
            Distribution="Student's T"
        ),
        name=f"{func_space}_p_val_ds"
    )
    workflow.connect([
        (beta_sources_node, p_val_ds, [
            ("sources", "Sources")
        ]),
        (regression_wf, p_val_ds, [
            ("outputnode.execute", "execute")
        ])
    ])

    # if all_opts.fir:
    combine_betas_node = Node(
        CombineFIRBetas(
            fir_count=all_opts.fir,
            volterra_count=all_opts.volterra_lag,
            brain_mask=all_opts.brain_mask,
        ),
        name=f"{func_space}_combine_fir_betas_node"
    )
    workflow.connect([
        (regression_wf, combine_betas_node, [
            ("outputnode.beta_files", "beta_files"),
            ("outputnode.beta_labels", "beta_labels"),
            ("outputnode.tstat_files", "tstat_files"),
            ("outputnode.pval_files", "pval_files"),
            ("outputnode.execute", "execute")
        ]),
        (combine_betas_node, beta_weights_ds, [
            ("beta_files", "in_file"),
            ("beta_labels", "condition"),
        ]),
        (combine_betas_node, t_stat_ds, [
            ("beta_files", "in_file"),
            ("beta_labels", "condition"),
        ]),
        (combine_betas_node, p_val_ds, [
            ("beta_files", "in_file"),
            ("beta_labels", "condition"),
        ]),
    ])

    if all_opts.save_intermediates:
        residual_bold_ds = Node(
            FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=need_compress,
                dismiss_entities=["run", "den"],
                desc="glmResidual",
                task=all_opts.task_rename,
                source_file=bold_runs_list
            ),
            name=f"{func_space}_residual_bold_ds"
        )
        residual_bold_sources_node = Node(
            BIDSSource(
                raw_dataset=all_opts.raw_bids,
                preprocessed_dataset=all_opts.preproc_bids,
                datasink_path=all_opts.datasink_path,
                work_path=all_opts.work
            ),
            name = "residual_bold_sources_node"
        )
        workflow.connect([
            (find_included_sources_node, residual_bold_sources_node, [
                ("bold_files", "file_1")
            ]),
            (design_ds, residual_bold_sources_node, [
                ("out_file", "file_2")
            ]),
            (residual_bold_sources_node, residual_bold_ds, [
                ("sources", "Sources")
            ]),
            (regression_wf, residual_bold_ds, [
                ("outputnode.bold_file", "in_file"),
                ("outputnode.execute", "execute")
            ]),
        ])

    r_squared_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="Rsquared",
            task=all_opts.task_rename,
            allowed_entities=("stat",),
            source_file=bold_runs_list
        ),
        name=f"{func_space}_r_squared_ds"
    )
    workflow.connect([
        (regression_wf, r_squared_ds, [
            ("outputnode.r_squared_file", "in_file"),
            ("outputnode.execute", "execute")
        ])
    ])

    mse_ds = Node(
        FLADataSink(
            base_directory=all_opts.datasink_path.parent,
            out_path_base=all_opts.datasink_path.name,
            compress=need_compress,
            extra_bids_patterns=all_opts.bids_patterns,
            dismiss_entities=["desc", "run", "den"],
            suffix="boldmap",
            stat="MSE",
            task=all_opts.task_rename,
            allowed_entities=("stat",),
            source_file=bold_runs_list
        ),
        name=f"{func_space}_mse_ds"
    )
    workflow.connect([
        (regression_wf, mse_ds, [
            ("outputnode.mse_file", "in_file"),
            ("outputnode.execute", "execute")
        ])
    ])

    return workflow


def build_run_workflow(run, task: str, func_space:str, file_extension: str):

    ### Define the workflow and the inputnode ###
    wf_name = f"task_{task}_run_{run}_processsing_wf"
    # logger.info(f"creating the run-level workflow: {wf_name}")
    workflow = Workflow(name=wf_name)
    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "bold_file",
                "tmask_file",
                "nuisance_design",
                "tr",
                "lh_surf",
                "rh_surf",
                "include"
            ]
        ),
        name="main_inputnode"
    )

    input_func_source_node = Node(
        IdentityInterface(
            fields=[
                "out_file"
            ]
        ),
        name="input_func_source_node"
    )
    workflow.connect([
        (inputnode, input_func_source_node, [
            ("bold_file", "out_file")
        ])
    ])

    outputnode = Node(
        IdentityInterface(
            fields=[
                "processed_bold",
            ]
        ),
        name="final_outputnode"
    )
    compress_files = file_extension.endswith(".gz")
    last_func_node = inputnode
    last_func_source_node = input_func_source_node

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
                ("include", "inputnode.execute")
            ]),
            (last_func_node, smoothing_wf, [
                ("bold_file", "inputnode.bold_file")
            ]),
        ])
        last_func_node = smoothing_wf.get_node("outputnode")

        if all_opts.save_intermediates:
            smoothed_ds = Node(FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="smooth",
                fwhm=f"{all_opts.fwhm:.2f} mm"
            ),
                name="smoothed_bold_ds"
            )
            smoothed_sources_node = Node(
                BIDSSource(
                    raw_dataset=all_opts.raw_bids,
                    preprocessed_dataset=all_opts.preproc_bids,
                    datasink_path=all_opts.datasink_path,
                    work_path=all_opts.work
                ),
                name="smoothed_bold_sources_node"
            )
            workflow.connect([
                (last_func_source_node, smoothed_sources_node, [
                    ("out_file", "file_1")
                ]),
                (inputnode, smoothed_sources_node, [
                    ("lh_surf", "file_2"),
                    ("rh_surf", "file_3")
                ]),
                (smoothed_sources_node, smoothed_ds, [
                    ("sources", "Sources")
                ]),
                (smoothing_wf, smoothed_ds, [
                    ("outputnode.bold_file", "in_file"),
                ]),
                (inputnode, smoothed_ds, [
                    ("bold_file", "source_file"),
                    ("include", "execute"),
                    ("tr", "RepetitionTime")
                ])
            ])
            last_func_source_node = smoothed_ds

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
            (inputnode, percent_change_node, [
                ("tmask_file", "tmask_in"),
                ("include", "execute")
            ])
        ])
        last_func_node = percent_change_node
        if all_opts.save_intermediates:
            percent_change_ds = Node(FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="percentChange",
            ),
                name="percent_change_bold_ds"
            )
            percent_change_sources_node = Node(
                BIDSSource(
                    raw_dataset=all_opts.raw_bids,
                    preprocessed_dataset=all_opts.preproc_bids,
                    datasink_path=all_opts.datasink_path,
                    work_path=all_opts.work
                ),
                name="percent_change_sources_node"
            )
            workflow.connect([
                (last_func_source_node, percent_change_sources_node, [
                    ("out_file", "file_1")
                ]),
                (inputnode, percent_change_sources_node, [
                    ("tmask_file", "file_2")
                ]),
                (percent_change_sources_node, percent_change_ds, [
                    ("sources", "Sources")
                ]),
                (percent_change_node, percent_change_ds, [
                    ("bold_file", "in_file")
                ]),
                (inputnode, percent_change_ds, [
                    ("bold_file", "source_file"),
                    ("include", "execute"),
                    ("tr", "RepetitionTime")
                ])
            ])
            last_func_source_node = percent_change_ds

    ### Nuisance regression ###
    if all_opts.nuisance_regression:

        regression_wf = build_regression_workflow(
            task=task, 
            run=run, 
            need_intercept=all_opts.exclude_run_mean
        )

        plot_nuisance_matrix_node = Node(
            PlotDesign(),
            name="plot_nuisance_design_node"
        )

        run_design_ds = Node(
            FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                dismiss_entities=["den"],
                desc="nuisance",
                space=func_space,
                suffix="design",
                task=task,
            ),
            name=f"task_{task}_run_{run}_design_ds"
        )

        workflow.connect([
            (last_func_node, regression_wf, [
                ("bold_file", "inputnode.bold_files")
            ]),
            (inputnode, regression_wf, [
                ("nuisance_design", "inputnode.design_matrices"),
                ("tmask_file", "inputnode.tmask_files"),
                ("include", "inputnode.execute")
            ]),
            (regression_wf, plot_nuisance_matrix_node, [
                ("outputnode.design_matrix", "design_matrix"),
                ("outputnode.execute", "execute")
            ]),
            (plot_nuisance_matrix_node, run_design_ds, [
                ("design_plot", "in_file")
            ]),
            (regression_wf, run_design_ds, [
                ("outputnode.execute", "execute")
            ]),
            (inputnode, run_design_ds, [
                ("bold_file", "source_file")
            ])
        ])
        last_func_node = regression_wf.get_node("outputnode")

        if all_opts.save_intermediates:
            regressed_bold_ds = Node(FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="nuisanceRegressed"
            ),
                name="regressed_bold_ds"
            )
            nuisance_regressed_sources_node = Node(
                BIDSSource(
                    raw_dataset=all_opts.raw_bids,
                    preprocessed_dataset=all_opts.preproc_bids,
                    datasink_path=all_opts.datasink_path,
                    work_path=all_opts.work
                ),
                name="nuisance_regressed_sources_node"
            )
            workflow.connect([
                (last_func_source_node, nuisance_regressed_sources_node, [
                    ("out_file", "file_1")
                ]),
                (inputnode, nuisance_regressed_sources_node, [
                    ("nuisance_design", "file_2"),
                    ("tmask_file", "file_3")
                ]),
                (nuisance_regressed_sources_node, regressed_bold_ds, [
                    ("sources", "Sources")
                ]),
                (regression_wf, regressed_bold_ds, [
                    ("outputnode.bold_file", "in_file")
                ]),
                (inputnode, regressed_bold_ds, [
                    ("bold_file", "source_file"),
                    ("include", "execute"),
                    ("tr", "RepetitionTime")
                ])
            ])
            combine_betas_node = Node(
                CombineFIRBetas(
                    fir_count=all_opts.fir,
                    volterra_count=all_opts.volterra_lag,
                    brain_mask=all_opts.brain_mask,
                ),
                name="combine_betas_node"
            )
            nuisance_betas_sources_node = Node(
                BIDSSource(
                    raw_dataset=all_opts.raw_bids,
                    preprocessed_dataset=all_opts.preproc_bids,
                    datasink_path=all_opts.datasink_path,
                    work_path=all_opts.work
                ),
                name="nuisance_betas_sources_node"
            )
            nuisance_betas_ds = Node(FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="nuisanceRegression",
                suffix="boldmap",
                stat="effect",
                allowed_entities=("condition", "stat"),
                Test="OLS"
            ),
                name="nuisance_betas_ds"
            )
            workflow.connect([
                (last_func_source_node, nuisance_betas_sources_node, [
                    ("out_file", "file_1")
                ]),
                (inputnode, nuisance_betas_sources_node, [
                    ("nuisance_design", "file_2"),
                    ("tmask_file", "file_3")
                ]),
                (regression_wf, combine_betas_node, [
                    ("outputnode.beta_files", "beta_files"),
                    ("outputnode.beta_labels", "beta_labels"),
                    ("outputnode.execute", "execute")
                ]),
                (nuisance_betas_sources_node, nuisance_betas_ds, [
                    ("sources", "Sources")
                ]),
                (combine_betas_node, nuisance_betas_ds, [
                    ("beta_files", "in_file"),
                    ("beta_labels", "condition")
                ]),
                (inputnode, nuisance_betas_ds, [
                    ("bold_file", "source_file"),
                    ("include", "execute")
                ])
            ])
            last_func_source_node = regressed_bold_ds

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
            (inputnode, filter_node, [
                ("tmask_file", "tmask_in"),
                ("tr", "tr"),
                ("include", "execute")
            ]),
        ])
        last_func_node = filter_node

        if all_opts.save_intermediates:
            filter_ds = Node(FLADataSink(
                base_directory=all_opts.datasink_path.parent,
                out_path_base=all_opts.datasink_path.name,
                extra_bids_patterns=all_opts.bids_patterns,
                compress=compress_files,
                dismiss_entities=["den"],
                desc="filtered",
            ),
                name="filtered_bold_ds"
            )
            filtered_sources_node = Node(
                BIDSSource(
                    raw_dataset=all_opts.raw_bids,
                    preprocessed_dataset=all_opts.preproc_bids,
                    datasink_path=all_opts.datasink_path,
                    work_path=all_opts.work
                ),
                name="filtered_sources_node"
            )
            if all_opts.highpass:
                filter_ds.inputs.Highpass = f"{all_opts.highpass:.3f} Hz"
            if all_opts.lowpass:
                filter_ds.inputs.Lowpass = f"{all_opts.lowpass:.3f} Hz"

            workflow.connect([
                (last_func_source_node, filtered_sources_node, [
                    ("out_file", "file_1"),
                ]),
                (inputnode, filtered_sources_node, [
                    ("tmask_file", "file_2")
                ]),
                (filtered_sources_node, filter_ds, [
                    ("sources", "Sources")
                ]),
                (filter_node, filter_ds, [
                    ("bold_file", "in_file")
                ]),
                (inputnode, filter_ds, [
                    ("bold_file", "source_file"),
                    ("include", "execute"),
                    ("tr", "RepetitionTime")
                ])
            ])
            last_func_source_node = filter_ds

    ### Connect final bold output ###
    if all_opts.save_intermediates:
        workflow.connect([
            (last_func_source_node, outputnode, [
                ("out_file", "processed_bold")
            ])
        ])
    else:
        workflow.connect([
            (last_func_node, outputnode, [
                ("bold_file", "processed_bold")
            ])
        ])

    return workflow


def build_regression_workflow(task:str, run=None, regression_columns=None, need_intercept=False):

    wf_label = f"task_{task}"
    if run:
        wf_label += f"_run_{run}"
    workflow = Workflow(name=f"{wf_label}_regression_wf")

    inputnode = Node(
        IdentityInterface(
            fields=[
                "bold_files",
                "design_matrices",
                "tmask_files",
                "inclusion_list",
                "execute"
            ]
        ),
        name="inputnode"
    )
    inputnode.inputs.execute = True

    outputnode = Node(
        IdentityInterface(
            fields=[
                "beta_files",
                "tstat_files",
                "pval_files",
                "beta_labels",
                "bold_file",
                "r_squared_file",
                "mse_file",
                "design_matrix",
                "unmasked_design_matrix",
                "tmask_file",
                "execute"
            ]
        ),
        name="outputnode"
    )

    concat_data_node = Node(
        ConcatRegressionData(
            include_intercept=need_intercept,
            task=task,
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
            ("design_matrices", "design_matrices_in"),
            ("inclusion_list", "inclusion_list"),
            ("execute", "execute")
        ]),
        (concat_data_node, glm_node, [
            ("bold_file", "bold_file_in"),
            ("design_matrix", "design_matrix"),
            ("tmask_file", "tmask_file"),
            ("execute", "execute")
        ]),
        (concat_data_node, outputnode, [
            ("design_matrix", "unmasked_design_matrix"),
            ("execute", "execute"),
            ("tmask_file", "tmask_file")
        ]),
        (glm_node, outputnode, [
            ("residual_bold_file", "bold_file"),
            ("masked_design_matrix", "design_matrix"),
            ("beta_files", "beta_files"),
            ("tstat_files", "tstat_files"),
            ("pval_files", "pval_files"),
            ("beta_labels", "beta_labels"),
            ("r_squared_file", "r_squared_file"),
            ("mse_file", "mse_file")
        ])
    ])

    if all_opts.fd_censoring:
        workflow.connect([
            (inputnode, concat_data_node, [
                ("tmask_files", "tmask_files_in")
            ])
        ])
    else:
        remove_motion_masking_node = Node(
            RemoveMotionMasking(
                start_censoring=all_opts.start_censoring
            ),
            name="remove_motion_masking_node"
        )
        workflow.connect([
            (inputnode, remove_motion_masking_node, [
                ("tmask_files", "tmask_file")
            ]),
            (remove_motion_masking_node, concat_data_node, [
                ("tmask_file", "tmask_files_in")
            ])
        ])

    return workflow


def build_exclusion_wf(run, task):

    workflow = Workflow(name=f"task_{task}_run_{run}_exclusion_wf")
    inputnode = Node(
        IdentityInterface(
            fields=[
                "subject",
                "session",
                "bold_file",
                "tmask_file"
            ]
        ),
        name="inputnode"
    )
    outputnode = Node(
        IdentityInterface(
            fields=[
                "include",
                "exclusion_table"
            ]
        ),
        name="outputnode"
    )
    
    validation_merging_node = Node(
        MergeUnique(),
        name="merge_validations_node"
    )

    ### Create tSNR node ###
    tsnr_check_node = Node(
        CheckRuntSNR(
            tsnr_threshold=all_opts.min_average_tsnr,
            brain_mask=all_opts.brain_mask,
        ),
        name="check_tsnr_node"
    )

    ### Create frame retention node ###
    frame_retention_check_node = Node(
        CheckRunRetention(
            minimum_frames=all_opts.min_run_frames,
            retention_threshold=all_opts.run_exclusion_threshold,
            start_censoring=all_opts.start_censoring
        ),
        name="check_frame_retention_node"
    )

    exclusion_table_node = Node(
        MakeRunExclusionTable(
            run=run,
            task=task,
            start_censoring=all_opts.start_censoring
        ),
        name="make_run_exclusion_table"
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
            ("minimum_frames_valid", "valid_x2"),
            ("retention_threshold_valid", "valid_x3")
        ]),
        (validation_merging_node, check_validation_node, [
            ("valid", "validation_list")
        ]),
        (check_validation_node, outputnode, [
            ("include", "include")
        ]),
        (frame_retention_check_node, exclusion_table_node, [
            ("minimum_frames_valid", "frame_minimum_valid"),
            ("retention_threshold_valid", "frame_retention_valid"),
            ("retention_percentage", "retention_percentage"),
            ("frames_retained", "frames_retained"),
            ("total_frames", "total_frames")
        ]),
        (tsnr_check_node, exclusion_table_node, [
            ("valid", "tsnr_valid"),
            ("whole_brain_tsnr", "whole_brain_tsnr")
        ]),
        (check_validation_node, exclusion_table_node, [
            ("include", "included")
        ]),
        (exclusion_table_node, outputnode, [
            ("exclusion_table", "exclusion_table")
        ])
    ])

    if all_opts.exclusion_file: 
        check_exclusion_file_node = Node(
            CheckExclusionFile(
                task=task,
                run=run,
                exclusion_file = all_opts.exclusion_file
            ),
            name="check_exclusion_file_node"
        )
        workflow.connect([
            (inputnode, check_exclusion_file_node, [
                ("subject", "subject"),
                ("session", "session")
            ]),
            (check_exclusion_file_node, validation_merging_node, [
                ("valid", "valid_x4")
            ]),
            (check_exclusion_file_node, exclusion_table_node, [
                ("valid", "pass_external_exclusion")
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
                "rh_surf",
                "execute"
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
            SurfaceSmooth(
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
                ("bold_file", "in_file"),
                ("lh_surf", "left_surf"),
                ("rh_surf", "right_surf"),
                ("execute", "execute")
            ]),
            (surface_smooth_node, outputnode, [
                ("out_file", "bold_file")
            ])
        ])

    elif is_nifti_file(file_extension):
        volume_smooth_node = Node(
            VolumeSmooth(
                kernel=fwhm2sigma(all_opts.fwhm),
            ),
            name="volume_smooth_node"
        )
        volume_smooth_node.inputs.environ.update(
            {"OMP_NUM_THREADS": str(all_opts.n_procs)})
        workflow.connect([
            (inputnode, volume_smooth_node, [
                ("bold_file", "volume_in"),
                ("execute", "execute")
            ]),
            (volume_smooth_node, outputnode, [
                ("volume_out", "bold_file")
            ])
        ])

    else:
        raise RuntimeError(f"oceanfla does not support smoothing for files of type <{file_extension}>")

    return workflow


def build_reporting_workflow(task:str, run:str=None):

    wf_label = f"task_{task}"
    if run:
        wf_label += f"_run_{run}"
    workflow = Workflow(name=f"{wf_label}_reporting_wf")

    inputnode = Node(
        IdentityInterface(
            fields=[
                "design_matrix",
                "final_tmask_file",
                "execute",
                "run_tmask_files",
                "confounds_files",
                "inclusion_list",
                "exclusion_tables"
            ],
            mandatory_inputs=False
        ),
        name="inputnode"
    )
    inputnode.inputs.execute = True

    outputnode = Node(
        IdentityInterface(
            fields=[
                "design_plot",
                "design_correlations",
                "exclusion_report"
            ],
            mandatory_inputs=False
        ),
        name="outputnode"
    )

    plot_design_node = Node(
        PlotDesign(),
        name="plot_design_node"
    )

    workflow.connect([
        (inputnode, plot_design_node, [
            ("design_matrix", "design_matrix"),
            ("execute", "execute")
        ]),
        (plot_design_node, outputnode, [
            ("design_plot", "design_plot"),
            ("design_correlations", "design_correlations")
        ])
    ])

    if run is None:
        report_exclusions_node = Node(
            ReportExclusions(
                task=task
            ),
            name="report_exclusions_node"
        )
        workflow.connect([
            (inputnode, report_exclusions_node, [
                ("exclusion_tables", "exclusion_tables"),
                ("run_tmask_files", "tmask_files"),
                ("confounds_files", "confounds_files"),
                ("inclusion_list", "inclusion_list"),
                ("final_tmask_file", "ses_tmask_file")
            ]), 
            (report_exclusions_node, outputnode, [
                ("exclusion_report", "exclusion_report")
            ])
        ])

    return workflow

    