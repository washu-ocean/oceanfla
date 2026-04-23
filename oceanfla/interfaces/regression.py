from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    traits,
)
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

    tstat_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )

    pval_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )

    beta_labels = traits.List(
        trait=traits.Str,
        desc=""
    )

    r_squared_file = traits.File(
        exists=True,
        desc=""
    ) 

    mse_file = traits.File(
        exists=True,
        desc=""
    ) 

    masked_design_matrix = traits.File(
        exists=True,
        desc="The design matrix for the regression after masking"
    )

    residual_bold_file = traits.File(
        exists=True,
        desc=""
    ) 


class RunGLMRegression(OptionalInterface):
    input_spec = RunGLMRegressionInputSpec
    output_spec = RunGLMRegressionOutputSpec

    def _run_interface(self, runtime):

        beta_files, tstat_files, pval_files, beta_labels, r_squared_file, mse_file, masked_design_file, func_residual_file = massuni_linGLM(
            func_file=self.inputs.bold_file_in,
            design_matrix=self.inputs.design_matrix,
            tmask_file=self.inputs.tmask_file,
            stdscale=self.inputs.stdscale,
            brain_mask=self.inputs.brain_mask
        )

        self._results["beta_files"] = beta_files
        self._results["tstat_files"] = tstat_files
        self._results["pval_files"] = pval_files
        self._results["beta_labels"] = beta_labels
        self._results["r_squared_file"] = r_squared_file
        self._results["mse_file"] = mse_file
        self._results["masked_design_matrix"] = masked_design_file
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
    task = traits.Str(
        desc="The task that this regression is for"
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
            task=self.inputs.task,
            brain_mask=self.inputs.brain_mask
        )

        return runtime


def massuni_linGLM(func_file: str,
                   design_matrix_file: str,
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
    from nilearn.glm.regression import OLSModel
    # from nilearn.glm.first_level import run_glm
    from scipy.stats import chi2, t

    func_data = load_data(func_file, brain_mask)
    design_matrix = pd.read_csv(design_matrix_file, sep="\t")
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

    # initialize the model and compute the betas
    model = OLSModel(masked_design_matrix)
    regression_results = model.fit(masked_func_data)

    # standardize the unmasked data
    if stdscale:
        func_data = func_ss.transform(func_data)
        design_matrix_data = design_ss.transform(design_matrix_data)

    # compute the residuals with unmasked data
    est_values = np.dot(design_matrix_data, regression_results.theta)
    resids = func_data - est_values

    # save the data out
    entities_base = {"desc": "modelOutput", "path": None}

    # beta files
    beta_files, beta_labels, = [], []
    wald_stat_files = []
    pval_files = []
    old_ext = parse_file_entities(func_file)["extension"]
    for i, beta_label in enumerate(design_matrix.columns):
        beta_entities = entities_base | {"suffix": f"beta-{beta_label}_boldmap"}
        t_stat_entities = entities_base | {"suffix": f"beta-{beta_label}-wald-t_boldmap"}
        student_p_entities = entities_base | {"suffix": f"beta-{beta_label}-2side-p_boldmap"}
        
        wald_t_stats = regression_results.t(i)
        student_p_two_sided = 2 * (1 - t.cdf(np.abs(wald_t_stats), regression_results.df_residuals))

        if is_cifti_file(func_file):
            beta_entities["ext"] = old_ext.replace("tseries", "scalar")
            t_stat_entities["ext"] = old_ext.replace("tseries", "scalar")
            student_p_entities["ext"] = old_ext.replace("tseries", "scalar")

        beta_filename = replace_entities(
            file=func_file,
            entities=beta_entities
        )
        create_image_like(
            data=(regression_results.theta[i])[np.newaxis, :],
            source_header=func_file,
            out_file=beta_filename,
            scalar_axis=[f"beta-{beta_label}"],
            brain_mask=brain_mask
        )
        beta_files.append(beta_filename)
        beta_labels.append(beta_label.replace("_", "-"))

        waldt_filename = replace_entities(
            file=func_file,
            entities=t_stat_entities
        )
        create_image_like(
            data=(wald_t_stats)[np.newaxis, :],
            source_header=func_file,
            out_file=waldt_filename,
            scalar_axis=[f"wald-t"],
            brain_mask=brain_mask
        )
        wald_stat_files.append(waldt_filename)

        pval_filename = replace_entities(
            file=func_file,
            entities=student_p_entities
        )
        create_image_like(
            data=(student_p_two_sided)[np.newaxis, :],
            source_header=func_file,
            out_file=pval_filename,
            scalar_axis=[f"p-val"],
            brain_mask=brain_mask
        )
        pval_files.append(pval_filename)

    # Model fit (R^2)
    r_squared_filename = replace_entities(
        file=func_file,
        entities=entities_base | {"suffix": "r-squared", "ext":old_ext.replace("tseries", "scalar")}
    )
    create_image_like(
        data=regression_results.r_square,
        source_header=func_file,
        out_file=r_squared_filename,
        scalar_axis=["r-squared"],
        brain_mask=brain_mask
    )

    # Mean square error
    mse_filename = replace_entities(
        file=func_file,
        entities=entities_base | {"suffix": "MSE", "ext":old_ext.replace("tseries", "scalar")}
    )
    create_image_like(
        data=regression_results.MSE,
        source_header=func_file,
        out_file=mse_filename,
        scalar_axis=["MSE"],
        brain_mask=brain_mask
    )

    # masked design matrix
    masked_design_file = replace_entities(
        file=design_matrix_file,
        entities=entities_base | {"suffix": "masked-design"}
    )
    used_design_matrix = design_matrix[mask]
    used_design_matrix.to_csv(masked_design_file, index=False, sep="\t")

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
    return (beta_files, wald_stat_files, pval_files, beta_labels, r_squared_filename, mse_filename, masked_design_file, residual_filename)


def combine_regression_data(task: str,
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

    entities_base = {"desc": "modelInput", "task": task, "path": None}
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


class CombineFIRBetasInputSpec(OptionalInterfaceSpec):
    beta_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )
    beta_labels = traits.List(
        trait=traits.Str,
        desc=""
    )
    brain_mask = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        usedefault=True,
        desc="The brain mask that accompanies volumetric data"
    )


class CombineFIRBetasOutputSpec(OptionalInterfaceSpec):
    beta_files = traits.List(
        trait=traits.File(exists=True),
        desc=""
    )
    beta_labels = traits.List(
        trait=traits.Str,
        desc=""
    )


class CombineFIRBetas(OptionalInterface):
    input_spec = CombineFIRBetasInputSpec
    output_spec = CombineFIRBetasOutputSpec

    def _run_interface(self, runtime):

        self._results["beta_files"], self._results["beta_labels"] = combine_fir_betas(
            beta_files=self.inputs.beta_files,
            beta_labels=self.inputs.beta_labels,
            brain_mask=self.inputs.brain_mask
        )

        return runtime
    

def combine_fir_betas(beta_files: list[str],
                      beta_labels: list[str],
                      brain_mask: str = None):
    import numpy as np
    from oceanfla.utilities import replace_entities, load_data, create_image_like, is_cifti_file
    from bids.layout import parse_file_entities
    
    label_file_map = {blabel: beta_files[i] for i, blabel in enumerate(beta_labels)}
    
    combo_label_file_map = {}
    for blabel in beta_labels:
        blabel_split = blabel.rsplit("-",1)
        if len(blabel_split) != 2:
            continue
        condition, fir_num = blabel_split[0], blabel_split[1]
        if fir_num.isnumeric() and len(fir_num)==2:
            if condition in combo_label_file_map:
                combo_label_file_map[condition].append(blabel)
            else:
                combo_label_file_map[condition] = [blabel]

    
    combo_beta_files, combo_beta_labels = [], []

    for condition, label_list in combo_label_file_map.items():
        sorted_bfiles = [label_file_map[blabel] for blabel in sorted(label_list, key=lambda x: int(x.rsplit("-",1)[-1]))]
        combined_beta_data = np.array([load_data(bf, brain_mask=brain_mask)[0,:] for bf in sorted_bfiles])
        new_entities = {"path":None, "beta":condition}
        if is_cifti_file(sorted_bfiles[0]):
            old_ext = parse_file_entities(sorted_bfiles[0])["extension"]
            new_entities["ext"] = old_ext.replace("scalar", "tseries")
        new_beta_file = replace_entities(
            file=sorted_bfiles[0],
            entities=new_entities
        )
        # scalar_axis_labels = ["fir-" + blabel.rsplit("-",1)[-1] for blabel in sorted(label_list, key=lambda x: int(x.rsplit("-",1)[-1]))]
        create_image_like(
            data=combined_beta_data,
            out_file=new_beta_file,
            source_header=sorted_bfiles[0],
            # scalar_axis=scalar_axis_labels,
            brain_mask=brain_mask
        )
        combo_beta_files.append(new_beta_file)
        combo_beta_labels.append(condition)

    return combo_beta_files, combo_beta_labels