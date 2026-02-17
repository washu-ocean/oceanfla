from nipype.interfaces.base import (
    SimpleInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec
)
from pathlib import Path
from nipype import Function


class EventsMatrixInputSpec(BaseInterfaceInputSpec):
    event_file = File(exists=True, mandatory=True,
                      desc="A BIDS style events file of type .tsv")

    fir = traits.Union(
        traits.Int(desc="An integer denoting the order of an FIR filter",
                   mandatory=False, allow_none=True),
        None,
        default_value=None)

    hrf = traits.Union(
        traits.List(trait=traits.Int, minlen=2, maxlen=2,
                    desc="A 2-element list, where hrf[0] denotes the time to the peak of an HRF, and hrf[1] denotes the duration of its 'undershoot' after the peak."),
        traits.File(exists=True),
        None,
        default_value=None)

    fir_vars = traits.Union(
        traits.List(trait=traits.Str,
                    desc="A list of column names denoting which columns should have an FIR filter applied."),
        None,
        default_value=None)

    hrf_vars = traits.Union(
        traits.List(trait=traits.Str,
                    desc="A list of column names denoting which columns should be convolved with the HRF function defined in the hrf list."),
        None,
        default_value=None)

    unmodeled = traits.Union(
        traits.List(trait=traits.Str,
                    desc="A list of column names denoting which columns should not be modeled by neither hrf or fir, but still included in the design matrix."),
        None,
        default_value=None)
    
    parameters = traits.Union(
        traits.List(trait=traits.Str,
                    desc="A list of parameter columns from the events file to include as parametric modulation regressors."),
        None,
        default_value=None)

    volumes = traits.Int(
        desc="The number of volumes that are in the corresponding BOLD run")

    tr = traits.Float(desc="The Repetition Time for this BOLD run")


class EventsMatrixOutputSpec(TraitedSpec):
    events_matrix = File(exists=True,
                         desc="A run-level design matrix created using the input parameters")


class EventsMatrix(SimpleInterface):
    input_spec = EventsMatrixInputSpec
    output_spec = EventsMatrixOutputSpec

    def _run_interface(self, runtime):

        self._results["events_matrix"] = make_design_matrix(
            event_file=self.inputs.event_file,
            tr=self.inputs.tr,
            volumes=self.inputs.volumes,
            fir=self.inputs.fir,
            fir_vars=self.inputs.fir_vars,
            hrf=self.inputs.hrf,
            hrf_vars=self.inputs.hrf_vars,
            unmodeled=self.inputs.unmodeled,
            parameters=self.inputs.parameters
        )

        return runtime


def make_design_matrix(event_file: str | Path,
                       volumes: int,
                       tr: float,
                       fir: int = None,
                       hrf: list[int] | str = None,
                       fir_vars: list[str] = None,
                       hrf_vars: list[str] = None,
                       unmodeled: list[str] = None,
                       parameters: list[str] = None):
    from oceanfla.utilities import replace_entities
    import pandas as pd
    import numpy as np
    from textwrap import dedent
    from oceanfla.config import get_logger
    logger = get_logger("nipype.interface")

    events_long = make_events_long(event_file, volumes, tr)
    events_matrix = events_long.copy()

    parametric_mod_regressors = make_parametric_modulation_regressors(event_file, parameters, volumes, tr) if parameters else None

    # If both FIR and HRF are specified, we should have at least one list
    # of columns for one of the categories specified.
    if (fir and hrf) and not (fir_vars or hrf_vars):
        raise RuntimeError(
            "Both FIR and HRF were specified, but you need to specify at least one list of variables (fir_vars or hrf_vars)")

    # fir_vars and hrf_vars must not have overlapping columns
    if (fir_vars and hrf_vars) and not set(fir_vars).isdisjoint(hrf_vars):
        raise RuntimeError(
            "Both FIR and HRF lists of variables were specified, but they overlap.")
    conditions = [s for s in np.unique(
        events_matrix.columns)]  # unique trial types
    residual_conditions = [
        c for c in conditions if c not in unmodeled] if unmodeled else conditions

    # Create other list if only one is specified
    if (fir and hrf) and (bool(fir_vars) ^ bool(hrf_vars)):
        if fir_vars:
            hrf_vars = [c for c in residual_conditions if c not in fir_vars]
        elif hrf_vars:
            fir_vars = [c for c in residual_conditions if c not in hrf_vars]
        assert set(hrf_vars).isdisjoint(fir_vars), f"Model conditions are not disjoint. hrf={hrf_vars},  fir={fir_vars}"

    if fir:
        fir_conditions = residual_conditions
        if fir_vars and len(fir_vars) > 0:
            fir_conditions = [c for c in residual_conditions if c in fir_vars]
        residual_conditions = [
            c for c in residual_conditions if c not in fir_conditions]

        col_names = {c: c + "-00" for c in fir_conditions}
        events_matrix = events_matrix.rename(columns=col_names)
        fir_cols_to_add = dict()
        for c in fir_conditions:
            for i in range(1, fir):
                fir_cols_to_add[f"{c}-{i:02d}"] = np.array(
                    np.roll(events_matrix.loc[:, col_names[c]], shift=i, axis=0))
                # so events do not roll back around to the beginnin
                fir_cols_to_add[f"{c}-{i:02d}"][:i] = 0
        events_matrix = pd.concat([events_matrix, pd.DataFrame(
            fir_cols_to_add, index=events_matrix.index)], axis=1)
        events_matrix = events_matrix.astype(int)

    if hrf:
        hrf_conditions = residual_conditions
        if hrf_vars and len(hrf_vars) > 0:
            hrf_conditions = [c for c in residual_conditions if c in hrf_vars]
        residual_conditions = [
            c for c in residual_conditions if c not in hrf_conditions]
        cfeats = hrf_convolve_features(features=events_matrix,
                                        column_names=hrf_conditions,
                                        time_col='index',
                                        units='s',
                                        time_to_peak=(hrf[0] if isinstance(
                                            hrf, list) else None),
                                        undershoot_dur=(
                                            hrf[1] if isinstance(hrf, list) else None),
                                        custom_hrf=(hrf if isinstance(hrf, str) else None))
        for c in hrf_conditions:
            events_matrix[c] = cfeats[c]
    
    events_matrix.sort_index(axis=1, inplace=True)

    if parametric_mod_regressors is not None:
        if isinstance(hrf, str):
            hrf_params = (None, None)
        elif isinstance(hrf, list):
            hrf_params = hrf
        else:
            hrf_params = (6, 12)
        convolved_parameters = hrf_convolve_features(features=parametric_mod_regressors,
                                                      column_names=parameters,
                                                      time_col='index',
                                                      units='s',
                                                      time_to_peak=hrf_params[0],
                                                      undershoot_dur=hrf_params[1],
                                                      custom_hrf=(hrf if isinstance(hrf, str) else None))
        # assert len(convolved_parameters) == len(events_matrix), "The length of the parametric modulation regressors is different than the length of the trial event regressors"
        for p in parameters:
            events_matrix.loc[:, f"{p}-modulator"] = convolved_parameters.loc[:, p]
    
    if len(residual_conditions) > 0:
        logger.warning(dedent(f"""The following trial types were not selected under either of the specified models
                        and were also not selected to be left unmodeled. These variables will not be included in the design matrix:\n\t {residual_conditions}"""))
        events_matrix = events_matrix.drop(columns=residual_conditions)

    out_file = replace_entities(
        event_file, {"suffix": "events-matrix", "ext": ".tsv", "path": None})
    events_matrix.to_csv(out_file, sep="\t", index=False)
    return out_file


def find_nearest(array, value):
    """
    Finds the smallest difference in 'value' and one of the
    elements of 'array', and returns the index of the element

    :param array: a list of elements to compare value to
    :type array: a list or list-like object
    :param value: a value to compare to elements of array
    :type value: integer or float
    :return: integer index of array
    :rtype: int
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx])


def make_events_long(event_file: Path, volumes: int, tr: float):
    """
    Takes and event file and a funtional run and creates a long formatted events file
    that maps the onset of task events to a frame of the functional run

    :param func_data: A numpy array-like object representing functional data
    :type bold_run: npt.ArrayLike
    :param event_file: path to the event timing file
    :type event_file: pathlib.Path
    :param tr: Repetition time of the function run in seconds
    :type tr: float
    :param output_file: file path (including name) to save the long formatted event file to
    :type output_file: pathlib.Path
    """
    import pandas as pd
    import numpy as np

    duration = tr * volumes
    events_df = pd.read_csv(event_file, sep="\t")
    conditions = [s for s in np.unique(events_df.trial_type)]
    events_long = pd.DataFrame(
        0, columns=conditions, index=np.arange(0, duration, tr)[:volumes])

    for e in events_df.index:
        i = find_nearest(events_long.index, events_df.loc[e, 'onset'])
        events_long.loc[i, events_df.loc[e, 'trial_type']] = 1
        if events_df.loc[e, 'duration'] > tr:
            offset = events_df.loc[e, 'onset'] + events_df.loc[e, 'duration']
            j = find_nearest(events_long.index, offset)
            events_long.loc[i:j, events_df.loc[e, 'trial_type']] = 1

    # if output_file and output_file.suffix == ".csv":
    #     logger.debug(f" saving events long to file: {output_file}")
    #     events_long.to_csv(output_file)

    return events_long


def make_parametric_modulation_regressors(event_file: str|Path,
                                          parameters: list[str],
                                          volumes: int, 
                                          tr: float):
    import pandas as pd
    import numpy as np

    duration = tr * volumes
    events_df = pd.read_csv(event_file, sep="\t")
    needed_cols = ["onset"] + parameters
    for nc in needed_cols: 
        assert nc in events_df.columns, f"cannot find the '{nc}' column in the events file: {event_file}"
    for p in parameters:
        orig_vals = events_df[p].to_numpy()
        demean_vals = orig_vals - np.nanmean(orig_vals)
        rescaled_vals = demean_vals / np.nanmax(np.abs(demean_vals))
        events_df[p] = rescaled_vals

    para_mod_regressors = pd.DataFrame(0, columns=parameters, index=np.arange(0, duration, tr)[:volumes])
    for e in events_df.index:
        i = find_nearest(para_mod_regressors.index, events_df.loc[e, "onset"])
        for p in parameters:
            para_mod_regressors.loc[i, p] = events_df.loc[e, p]
    para_mod_regressors = para_mod_regressors.fillna(0)
    para_mod_regressors.to_csv("paramod_regressors.csv")
    return para_mod_regressors
    

def hrf_convolve_features(features,
                           column_names: list = None,
                           time_col: str = 'index',
                           units: str = 's',
                           time_to_peak: int = 5,
                           undershoot_dur: int = 12,
                           custom_hrf: Path = None):
    """
    This function convolves a hemodynamic response function with each column in a timeseries dataframe.

    Parameters
    ----------
    features: DataFrame
        A Pandas dataframe with the feature signals to convolve.
    column_names: list
        List of columns names to use; if it is None, use all columns. Default is None.
    time_col: str
        The name of the time column to use if not the index. Default is "index".
    units: str
        Must be 'ms','s','m', or 'h' to denote milliseconds, seconds, minutes, or hours respectively.
    time_to_peak: int
        Time to peak for HRF model. Default is 5 seconds.
    undershoot_dur: int
        Undershoot duration for HRF model. Default is 12 seconds.

    Returns
    -------
    convolved_features: DataFrame
        The HRF-convolved feature timeseries
    """
    import pandas as pd
    import numpy as np

    if not column_names:
        column_names = features.columns

    if time_col == 'index':
        time = features.index.to_numpy()
    else:
        time = features[time_col]
        features.index = time

    if units == 'm' or units == 'minutes':
        features.index = features.index * 60
        time = features.index.to_numpy()
    if units == 'h' or units == 'hours':
        features.index = features.index * 3600
        time = features.index.to_numpy()
    if units == 'ms' or units == 'milliseconds':
        features.index = features.index / 1000
        time = features.index.to_numpy()

    convolved_features = pd.DataFrame(index=time)
    hrf_sig = np.loadtxt(custom_hrf) if custom_hrf is not None else create_hrf(
        time, time_to_peak=time_to_peak, undershoot_dur=undershoot_dur)

    for a in column_names:
        convolved_features[a] = np.convolve(features[a], hrf_sig)[:len(time)]

    return convolved_features


def create_hrf(time, time_to_peak=5, undershoot_dur=12):
    from scipy.stats import gamma
    """
    This function creates a hemodynamic response function timeseries.

    Parameters
    ----------
    time: numpy array
        a 1D numpy array that makes up the x-axis (time) of our HRF in seconds
    time_to_peak: int
        Time to HRF peak in seconds. Default is 5 seconds.
    undershoot_dur: int
        Duration of the post-peak undershoot. Default is 12 seconds.

    Returns
    -------
    hrf_timeseries: numpy array
        The y-values for the HRF at each time point
    """
    peak = gamma.pdf(time, time_to_peak)
    undershoot = gamma.pdf(time, undershoot_dur)
    hrf_timeseries = peak - 0.35 * undershoot
    return hrf_timeseries


def get_number_of_volumes(bold_in, brain_mask=None):
    from oceanfla.utilities import load_data
    func_data = load_data(func_file=bold_in,
                          brain_mask=brain_mask)
    return func_data.shape[0]


GetVolumeCount = Function(
    function=get_number_of_volumes,
    input_names=["bold_in", "brain_mask"],
    output_names=["volumes"]
)


class ModifyEventsFileInputSpec(BaseInterfaceInputSpec):
    events_file = File(
        exists=True,
        mandatory=True,
        desc="A BIDS style events file of type .tsv"
    )
    trial_type_map = traits.Union(
        traits.DictStrStr(),
        None,
        desc="A dictionary mapping 'trial_type' names in the event file to new names"
    )
    removal_list = traits.Union(
        traits.List(
            item_trait=traits.Str
        ),
        None,
        default_value=None,
        desc="A list of 'trial_type' values to remove from the file"
    )


class ModifyEventsFileOutputSpec(TraitedSpec):
    events_out = File(
        exists=True,
        desc="the modified events file"
    )


class ModifyEventsFile(SimpleInterface):
    input_spec = ModifyEventsFileInputSpec
    output_spec = ModifyEventsFileOutputSpec

    def _run_interface(self, runtime):

        self._results["events_out"] = modify_events_file(
            events_file=self.inputs.events_file,
            trial_type_map=self.inputs.trial_type_map,
            removal_list=self.inputs.removal_list
        )

        return runtime


def modify_events_file(events_file: str | Path,
                       trial_type_map: dict,
                       removal_list: list[str]):
    import pandas as pd
    from oceanfla.utilities import replace_entities

    events_df = pd.read_csv(events_file, sep="\t")

    # rename the trial types
    if trial_type_map:
        events_df['trial_type'] = events_df["trial_type"].replace(trial_type_map)

    # remove trials of a certain type
    if removal_list:
        events_df = events_df[~events_df["trial_type"].isin(removal_list)]

    out_file = replace_entities(
        events_file, {"suffix": "modified-events", "ext": ".tsv", "path": None})
    events_df.to_csv(out_file, sep="\t", index=False)
    return out_file
