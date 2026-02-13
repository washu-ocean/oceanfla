from matplotlib import axes
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from oceanfla.interfaces.utility import (
    OptionalInterface,
    OptionalInterfaceSpec
)

class PlotDesignInputSpec(OptionalInterfaceSpec):
    design_matrix = traits.Union(
        traits.File(exists=True),
        None,
        desc="The design matrix to plot"
    )
    tmask_file = traits.Union(
        traits.File(exists=True),
        None,
        desc="The temporal mask file",
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
            design_matrix=self.inputs.design_matrix,
            tmask_file=self.inputs.tmask_file
        )
        return runtime
    

def plot_design_matrix(design_matrix, tmask_file=None):
    from oceanfla.utilities import replace_entities
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_design_matrix, plot_design_matrix_correlation


    design_df = pd.read_csv(design_matrix, sep="\t")
    mask = np.loadtxt(tmask_file).astype(bool) if tmask_file else np.full(
        shape=(len(design_df),), fill_value=True)
    
    masked_design_df = design_df[mask]
    num_conditions = len(masked_design_df.columns)
    num_rows = len(masked_design_df)

    dmat_grid_rows = num_rows//10
    fig_width, fig_height = num_conditions, (dmat_grid_rows+ num_conditions)
    
    design_plot_file = replace_entities(
        file=design_matrix,
        entities={"ext": ".png", "path": None}
    )
    design_corr_file = replace_entities(
        file=design_matrix,
        entities={"ext": ".png", "path": None, "suffix":"design-corr"}
    )

    plot_design_matrix(masked_design_df, 
                       output_file=design_plot_file)

    plot_design_matrix_correlation(masked_design_df, 
                                   output_file=design_corr_file, 
                                   title="Condition Correlations")

    # fig.savefig(design_plot_file, bbox_inches="tight")
    return design_plot_file, design_corr_file