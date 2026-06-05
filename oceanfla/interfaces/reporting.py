from nipype.interfaces.base import (
    traits,
    SimpleInterface,
    BaseInterfaceInputSpec,
    TraitedSpec
)
from oceanfla.interfaces.utility import (
    OptionalInterface,
    OptionalInterfaceSpec
)

class PlotDesignInputSpec(OptionalInterfaceSpec):
    design_matrix = traits.Union(
        traits.File(exists=True),
        None,
        desc="The masked design matrix to plot"
    )

class PlotDesignOutputSpec(OptionalInterfaceSpec):
    design_plot = traits.File(
        exists=True,
        desc="A saved png plot of the design matrix"
    )
    design_correlations = traits.File(
        exists=True,
        desc="A saved png plot of condition correlations"
    )

class PlotDesign(OptionalInterface):
    input_spec = PlotDesignInputSpec
    output_spec = PlotDesignOutputSpec

    def _run_interface(self, runtime):
        self._results["design_plot"], self._results["design_correlations"] = plot_design_matrix(
            design_matrix=self.inputs.design_matrix
        )
        return runtime
    

def plot_design_matrix(design_matrix):
    from oceanfla.utilities import replace_entities
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_design_matrix, plot_design_matrix_correlation


    design_df = pd.read_csv(design_matrix, sep="\t")
    # mask = np.loadtxt(tmask_file).astype(bool) if tmask_file else np.full(
    #     shape=(len(design_df),), fill_value=True)
    
    # masked_design_df = design_df[mask]
    # num_conditions = len(masked_design_df.columns)
    # num_rows = len(masked_design_df)

    # dmat_grid_rows = num_rows//10
    # fig_width, fig_height = num_conditions, (dmat_grid_rows+ num_conditions)
    
    design_plot_file = replace_entities(
        file=design_matrix,
        entities={"ext": ".png", "path": None}
    )
    design_corr_file = replace_entities(
        file=design_matrix,
        entities={"ext": ".png", "path": None, "suffix":"design-corr"}
    )

    dm_plot_ax = plot_design_matrix(design_df)
    dm_plot_fig = dm_plot_ax.get_figure()
    dm_plot_fig.savefig(design_plot_file, dpi=300)
    plt.close()

    dm_corr_ax = plot_design_matrix_correlation(design_df,
                                                title="Condition Correlations")
    dm_corr_fig = dm_corr_ax.get_figure()
    dm_corr_fig.savefig(design_corr_file, dpi=300)
    plt.close()

    # fig.savefig(design_plot_file, bbox_inches="tight")
    return design_plot_file, design_corr_file


class ReportExclusionsInputSpec(BaseInterfaceInputSpec):
    task = traits.Str()
    exclusion_tables = traits.List(
        trait=traits.File(exists=True),
        desc="List of run-level exclusion tables"
    )
    tmask_files = traits.List(
        trait=traits.File(exists=True),
        desc="List of run-level tmask_files"
    )
    confounds_files = traits.List(
        trait=traits.File(exists=True),
        desc="List of run-level confounds files"
    )
    inclusion_list = traits.List(
        trait=traits.Bool,
        desc="A list of boolean values to indicate inclusion in the final concatenated data"
    )
    ses_tmask_file = traits.Union(
        None,
        traits.File(exists=True),
        desc="The session-level tmask file",
    )


class ReportExclusionsOutputSpec(TraitedSpec):
    exclusion_report = traits.File(
        exists=True,
        desc="Table containing exclusion information across all runs"
    )

class ReportExclusions(SimpleInterface):
    input_spec = ReportExclusionsInputSpec
    output_spec = ReportExclusionsOutputSpec

    def _run_interface(self, runtime):
        
        self._results["exclusion_report"] = make_exclusion_report(
            task=self.inputs.task,
            exclusion_tables=self.inputs.exclusion_tables,
            tmask_files=self.inputs.tmask_files,
            confounds_files=self.inputs.confounds_files,
            inclusion_list=self.inputs.inclusion_list,
            ses_tmask=self.inputs.ses_tmask_file
        )
        return runtime
    

def make_exclusion_report(task: str,
                          exclusion_tables: list,
                          tmask_files: list,
                          confounds_files: list,
                          inclusion_list: list[bool],
                          ses_tmask: str=None):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    lengths = [len(x) for x in 
               [exclusion_tables, tmask_files, confounds_files, inclusion_list]]
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"All input lists must be the same length: {lengths}")
    
    exclusion_df_list = []
    ses_fd_list = []
    total_frame_count = 0
    total_non_censored_frames = 0
    
    for i in range(lengths[0]):
        excl_df = pd.read_csv(exclusion_tables[i])
        fd = pd.read_csv(confounds_files[i], sep="\t")["framewise_displacement"].to_numpy()
        tmask = np.loadtxt(tmask_files[i]).astype(bool)
        avg_fd = np.nanmean(fd)
        avg_masked_fd = np.nanmean(fd[tmask])
        excl_df.loc[0, "Avg FD (mm)"] = avg_fd
        excl_df.loc[0, "Avg FD after tmask (mm)"] = avg_masked_fd
        exclusion_df_list.append(excl_df)
        if inclusion_list[i]:
            ses_fd_list.append(fd)
            run_tot_frames = int(excl_df.loc[0, "total frames"])
            total_frame_count += run_tot_frames
            total_non_censored_frames += int(excl_df.loc[0, "frames after start censoring"])
            if fd.shape[0] != excl_df.loc[0, "total frames"]:
                raise RuntimeError(
                    f"Length of confounds and reported frames is different for run-{excl_df.loc[0, 'run']}"
                )
            

    # combine the run-level exclusion tables
    exclusion_report = pd.concat(exclusion_df_list, axis=0, ignore_index=True)
    exclusion_report.sort_values(by=["task", "run"], inplace=True)
    exclusion_report["run"] = exclusion_report["run"].astype(str).apply(lambda x: f"{int(x):02d}")
    for col in ["total frames", "frames after start censoring", "frames retained"]:
        exclusion_report[col] = exclusion_report[col].astype(int)
    exclusion_report = exclusion_report.reset_index(drop=True)

    # compute session-level stats
    if len(ses_fd_list) > 0 and total_frame_count > 0:
        ses_index = len(exclusion_report)
        exclusion_report.loc[ses_index, "task"] = task
        exclusion_report.loc[ses_index, "run"] = "ses"

        ses_fd = np.concatenate(ses_fd_list, axis=0)
        if ses_fd.shape[0] != total_frame_count:
            raise RuntimeError(
                f"Length of confounds and reported frames is different"
            )
        avg_ses_fd = np.nanmean(ses_fd)
        exclusion_report.loc[ses_index, "Avg FD (mm)"] = avg_ses_fd
        if not ses_tmask:
            raise ValueError(
                "Session-level GLM was run, but no session-level tmask was provided"
            )
        ses_level_tmask = np.loadtxt(ses_tmask).astype(bool)
        if ses_level_tmask.shape[0] != total_frame_count:
            raise RuntimeError(
                f"Length of session tmask and reported frames is different"
            )
        avg_ses_masked_fd = np.nanmean(ses_fd[ses_level_tmask])
        exclusion_report.loc[ses_index, "Avg FD after tmask (mm)"] = avg_ses_masked_fd

        ses_frames_retained = np.nansum(ses_level_tmask)
        ses_perc_frames_retained = (ses_frames_retained / total_non_censored_frames) * 100
        exclusion_report.loc[ses_index, "total frames"] = int(total_frame_count)
        exclusion_report.loc[ses_index, "frames after start censoring"] = int(total_non_censored_frames)
        exclusion_report.loc[ses_index, "frames retained"] = int(ses_frames_retained)
        exclusion_report.loc[ses_index, "% frames retained"] = ses_perc_frames_retained

    outfile = f"task-{task}_desc-ses-level-exclusion_report.csv"
    outfile = str(Path().resolve() / outfile)
    exclusion_report.to_csv(outfile, index=False)
    return outfile

    
        
