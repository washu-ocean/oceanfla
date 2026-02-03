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
            design_matrix=self.inputs.design_matrix
        )
        return runtime
    

def plot_design_matrix(design_matrix):
    pass