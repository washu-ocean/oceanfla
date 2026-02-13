from click import option
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from sqlalchemy import func
from oceanfla.interfaces.utility import ( 
    OptionalInterface, 
    OptionalInterfaceSpec,
)

class RunGLMRegressionInputSpec(OptionalInterfaceSpec):
    bold_file_in = traits.Union(
        traits.File(exists=True),
        None,
        desc="The functional data file"
    )
    design_matrix = traits.Union(
        traits.File(exists=True),
        None,
        desc="The design matrix for the regression"
    )
    tmask_file = traits.Union(
        traits.File(exists=True),
        None,
        desc="The temporal mask file",
    )
    stdscale = traits.Bool(
        default_value=False,
        desc="Flag to indicate standard scaling the data before the regression"
    )
    brain_mask = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        usedefault=True,
        desc="The brain mask that accompanies volumetric data"
    )


class RunGLMRegressionOutputSpec(OptionalInterfaceSpec):
    beta_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )

    beta_labels = traits.List(
        trait=traits.Str,
        desc=""
    )

    residual_bold_file = traits.File(
        exists=True,
        desc="")


class RunGLMRegression(OptionalInterface):
    input_spec = RunGLMRegressionInputSpec
    output_spec = RunGLMRegressionOutputSpec

    def _run_interface(self, runtime):

        beta_files, beta_labels, func_residual_file = massuni_linGLM(
            func_file=self.inputs.bold_file_in,
            design_matrix=self.inputs.design_matrix,
            tmask_file=self.inputs.tmask_file,
            stdscale=self.inputs.stdscale,
            brain_mask=self.inputs.brain_mask
        )

        self._results["beta_files"] = beta_files
        self._results["beta_labels"] = beta_labels
        self._results["residual_bold_file"] = func_residual_file

        return runtime


class ConcatRegressionDataInputSpec(OptionalInterfaceSpec):
    bold_files_in = traits.Union(
        traits.List(),
        traits.File(exists=True),
        desc="A list of functional data files"
    )
    design_matrices_in = traits.Union(
        traits.List(),
        traits.File(exists=True),
        desc="A list of event matrix files"
    )
    tmask_files_in = event_matrices = traits.Union(
        traits.List(),
        None,
        traits.File(exists=True),
        desc="A list of temporal mask files "
    )
    inclusion_list = traits.List(
        trait=traits.Bool,
        desc="A list of boolean values to indicate inclusion in the final concatenated data"
    )
    include_intercept = traits.Bool(
        default_value=True,
        desc=""
    )
    tasks = traits.Union(
        traits.List(trait=traits.Str),
        traits.Str,
        desc="The task(s) that this regression is for"
    )
    brain_mask = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )


class ConcatRegressionDataOutputSpec(OptionalInterfaceSpec):
    bold_file = traits.Union(
        None,
        traits.File(exists=True),
        desc=""
    )
    design_matrix = traits.Union(
        None,
        traits.File(exists=True),
        desc=""
    )
    tmask_file = traits.Union(
        None,
        traits.File(exists=True),
        desc="The concatenation of the input list 'tmask_files_in'"
    )


class ConcatRegressionData(OptionalInterface):
    input_spec = ConcatRegressionDataInputSpec
    output_spec = ConcatRegressionDataOutputSpec

    def _run_interface(self, runtime):
        from bids.utils import listify

        func_files = listify(self.inputs.bold_files_in)
        design_matrices = listify(self.inputs.design_matrices_in)
        tmask_files = listify(self.inputs.tmask_files_in)

        self._results["bold_file"], self._results["design_matrix"], self._results["tmask_file"], self._results["execute"] = combine_regression_data(
            func_list=func_files,
            design_matrix_files=design_matrices,
            tmask_files=tmask_files,
            inclusion_list=self.inputs.inclusion_list,
            add_intercept=self.inputs.include_intercept,
            tasks=self.inputs.tasks,
            brain_mask=self.inputs.brain_mask
        )

        return runtime


def massuni_linGLM(func_file: str,
                   design_matrix: str,
                   tmask_file: str,
                   stdscale: bool,
                   brain_mask: str = None):
    """
    Compute the mass univariate GLM.

    Parameters
    ----------

    func_data: npt.ArrayLike
        Numpy array representing BOLD data
    design_matrix: pd.DataFrame
        DataFrame representing a design matrix for the GLM
    mask: npt.ArrayLike
        Numpy array representing a mask to apply to the two other parameters
    """
    from oceanfla.utilities import load_data, create_image_like, replace_entities, is_cifti_file
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from bids.layout import parse_file_entities

    func_data = load_data(func_file, brain_mask)
    design_matrix = pd.read_csv(design_matrix, sep="\t")
    design_matrix_data = design_matrix.to_numpy()
    mask = np.loadtxt(tmask_file).astype(bool) if tmask_file else np.full(
        shape=(func_data.shape[0],), fill_value=True)

    assert design_matrix_data.shape[0] == func_data.shape[
        0], "the design matrix must be the same length as the functional data"
    assert mask.shape[0] == func_data.shape[0], "the mask must be the same length as the functional data"

    # apply the mask to the data
    masked_func_data = func_data.copy()[mask, :]
    masked_design_matrix = design_matrix_data.copy()[mask, :]

    # standardize the masked data
    func_ss = StandardScaler()
    design_ss = StandardScaler()
    if stdscale:
        masked_func_data = func_ss.fit_transform(masked_func_data)
        masked_design_matrix = design_ss.fit_transform(masked_design_matrix)

    # compute beta values
    inv_mat = np.linalg.pinv(masked_design_matrix)
    beta_data = np.dot(inv_mat, masked_func_data)

    # standardize the unmasked data
    if stdscale:
        func_data = func_ss.transform(func_data)
        design_matrix_data = design_ss.transform(design_matrix_data)

    # compute the residuals with unmasked data
    est_values = np.dot(design_matrix_data, beta_data)
    resids = func_data - est_values

    # save the data out
    entities_base = {"desc": "modelOutput", "path": None}

    # beta files
    beta_files, beta_labels, = [], []
    for i, beta_label in enumerate(design_matrix.columns):
        beta_entities = entities_base | {"suffix": f"beta-{beta_label}_boldmap"}
        if is_cifti_file(func_file):
            old_ext = parse_file_entities(func_file)["extension"]
            beta_entities["ext"] = old_ext.replace("tseries", "scalar")
        beta_filename = replace_entities(
            file=func_file,
            entities=beta_entities
        )
        create_image_like(
            data=(beta_data[i])[np.newaxis, :],
            source_header=func_file,
            out_file=beta_filename,
            dscalar_axis=[f"beta-{beta_label}"],
            brain_mask=brain_mask
        )
        beta_files.append(beta_filename)
        beta_labels.append(beta_label.replace("_", "-"))

    # residual bold data
    residual_filename = replace_entities(
        file=func_file,
        entities=entities_base | {"suffix": "residual-bold"}
    )
    create_image_like(
        data=resids,
        source_header=func_file,
        out_file=residual_filename,
        brain_mask=brain_mask
    )
    return (beta_files, beta_labels, residual_filename)


def combine_regression_data(tasks: list,
                            func_list: list=None,
                            tmask_files: list = None,
                            design_matrix_files: list = None,
                            inclusion_list: list[bool] = None,
                            add_intercept=True,
                            brain_mask: str = None):
    from oceanfla.utilities import replace_entities, load_data, create_image_like
    import numpy as np
    import pandas as pd

    lengths = [len(x) for x in
               [func_list, tmask_files, design_matrix_files, inclusion_list]
               if x]
    if not len(set(lengths)) == 1:
        raise RuntimeError(
            f"All input lists must be the same length: {lengths}")

    # remove any runs that are being excluded
    if inclusion_list:
        if not any(inclusion_list):
            res_list = [None for data_list in [func_list, tmask_files, design_matrix_files]] + [False]
            return res_list
        
        remaining_data_lists = [[x for i, x in enumerate(data_list) if inclusion_list[i]]
                                if data_list else None for data_list in
                                [func_list, tmask_files, design_matrix_files]]
        func_list, tmask_files, design_matrix_files = remaining_data_lists
    
    
    func_data_list = [load_data(f, brain_mask) for f in func_list]
    design_matrices = [pd.read_csv(
        f, sep="\t") for f in design_matrix_files]
    tmask_list = [np.loadtxt(f) for f in tmask_files] if tmask_files else None

    # concatenate all of the data on the time axis
    task_label = "-".join(tasks)
    entities_base = {"desc": "modelInput", "task": task_label, "path": None}
    if len(func_list) > 1:
        entities_base["run"] = None
    res_list = []

    # functional data
    final_func_file = replace_entities(
        file=func_list[0],
        entities=entities_base
    )
    create_image_like(
        data=np.concatenate(func_data_list, axis=0),
        source_header=func_list[0],
        out_file=final_func_file,
        brain_mask=brain_mask)
    res_list.append(final_func_file)

    # design data
    final_design = pd.concat(design_matrices, axis=0,
                             ignore_index=True).fillna(0)
    if add_intercept:
        final_design.loc[:, "intercept"] = 1

    final_design_file = replace_entities(
        file=design_matrix_files[0],
        entities=entities_base | {"suffix": "design"}
    )
    final_design.to_csv(final_design_file, index=False, sep="\t")

    res_list.append(final_design_file)

    # tmask data
    final_tmask = None
    if tmask_list is not None:
        for i, run_tm in enumerate(tmask_list):
            if run_tm.shape[0] != func_data_list[i].shape[0]:
                raise RuntimeError(
                    f"Grouped functional data and tmasks must all have the same number of timepoints, but don't: {func_data_list[i].shape[0]}, {run_tm.shape[0]}")
        final_tmask = replace_entities(
            file=tmask_files[0], entities=entities_base)
        np.savetxt(final_tmask,
                   np.concatenate(tmask_list, axis=0))
    res_list.append(final_tmask)

    res_list.append(True)
    return res_list


class MakeRunDesignInputSpec(BaseInterfaceInputSpec):
    event_matrix = traits.File(
        exists=True,
        desc="An event matrix file"
    )
    nuisance_matrix = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        desc="An nuisance matrix file"
    )
    nuisance_regressors = traits.Union(
        traits.List(trait=traits.Str),
        None,
        default_value=None,
        desc="A list of column names to be used for nuisance regression."
    )


class MakeRunDesignOutputSpec(TraitedSpec):
    nuisance_design = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        desc="The run-level design matrix for nuisance regression"
    )
    main_design = traits.File(
        exists=True,
        desc="The run-level design matrix for main effect regression"
    )


class MakeRunDesign(SimpleInterface):
    input_spec = MakeRunDesignInputSpec
    output_spec = MakeRunDesignOutputSpec

    def _run_interface(self, runtime):

        self._results["main_design"], self._results["nuisance_design"] = make_run_design_files(
            event_matrix=self.inputs.event_matrix,
            nuisance_matrix=self.inputs.nuisance_matrix,
            nuisance_regressors=self.inputs.nuisance_regressors
        )
        return runtime


def make_run_design_files(event_matrix: str,
                          nuisance_matrix: str = None,
                          nuisance_regressors: list[str] = None):
    import pandas as pd
    import numpy as np
    from oceanfla.utilities import replace_entities

    event_df = pd.read_csv(event_matrix, sep="\t")
    main_design_file = replace_entities(
        event_matrix, 
        {"suffix": "main-design", "ext": ".tsv", "path": None}
    )
    nuisance_design_file = replace_entities(
        nuisance_matrix,
        {"suffix": "nuisance-design", "ext": ".tsv", "path": None}
    )
    
    combo_df = None
    if not nuisance_matrix:
        if not nuisance_regressors:
            return event_matrix, None
        else:
            combo_df = event_df
    else:
        nuisance_df = pd.read_csv(nuisance_matrix, sep="\t")
        combo_df = pd.concat([event_df.reset_index(drop=True), 
                                nuisance_df.reset_index(drop=True)], 
                              axis=1)
        if not nuisance_regressors:
            combo_df.to_csv(main_design_file, sep="\t", index=False)
            return main_design_file, None
    
    all_design_columns = combo_df.columns.to_list()
    main_regressors = [
        dc for dc in all_design_columns if dc not in nuisance_regressors]
    if len(main_regressors) < 1:
        raise ValueError("All regressor columns are being used for nuisance regression")
    
    nuisance_design = combo_df.loc[:, nuisance_regressors]
    main_design = combo_df.loc[:, main_regressors]

    main_design.to_csv(main_design_file, sep="\t", index=False)
    nuisance_design.to_csv(nuisance_design_file, sep="\t", index=False)

    return main_design_file, nuisance_design_file