from matplotlib import axes
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)

class PlotDesignInputSpec(BaseInterfaceInputSpec):
    design_matrix = traits.File(
        exists=True,
        mandatory=True,
        desc="The design matrix to plot"
    )
    tmask_file = traits.Union(
        traits.File(exists=True),
        None,
        desc="The temporal mask file",
    )

class PlotDesignOutputSpec(TraitedSpec):
    design_plot = traits.File(
        exists=True,
        desc="A saved png plot of the design matrix"
    )

class PlotDesign(SimpleInterface):
    input_spec = PlotDesignInputSpec
    output_spec = PlotDesignOutputSpec

    def _run_interface(self, runtime):
        self._results["design_plot"] = plot_design_matrix(
            design_matrix=self.inputs.design_matrix,
            tmask_file=self.inputs.tmask_file
        )
        return runtime
    

def plot_design_matrix(design_matrix, tmask_file=None):
    from oceanfla.utilities import replace_entities
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from nilearn.plotting import plot_design_matrix, plot_design_matrix_correlation

    design_df = pd.read_csv(design_matrix, sep="\t")
    mask = np.loadtxt(tmask_file).astype(bool) if tmask_file else np.full(
        shape=(len(design_df),), fill_value=True)
    
    masked_design_df = design_df[mask]
    num_conditions = len(masked_design_df.columns)
    num_rows = len(masked_design_df)

    dmat_grid_rows = num_rows//10
    fig_width, fig_height = num_conditions, (dmat_grid_rows+ num_conditions)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(nrows=(dmat_grid_rows + num_conditions), ncols=num_conditions, figure=fig)
    
    # plot the design matrix
    ax1 = fig.add_subplot(gs[:dmat_grid_rows, :])
    plot_design_matrix(masked_design_df, axes=ax1, fontsize=20)
    ax1.set_title("Final Design Matrix", fontsize=20)

    # plot the 
    ax2 = fig.add_subplot(gs[dmat_grid_rows:, :])
    plot_design_matrix_correlation(masked_design_df, axes=ax2, fontsize=20)
    ax2.set_title("Condition Correlations", fontsize=20)


    design_plot_file = replace_entities(
        file=design_matrix,
        entities={"ext": ".png", "path": None}
    )
    fig.savefig(design_plot_file, bbox_inches="tight")
    return design_plot_file