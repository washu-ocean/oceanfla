from pathlib import Path
from networkx import desargues_graph
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from sqlalchemy import desc


class CheckRunRetentionInputSpec(BaseInterfaceInputSpec):
    tmask_file = File(
        exists=True,
        mandatory=True,
        desc="Path to the tmask file"
    )

    retention_threshold = traits.Float(
        default_value=0.0,
        desc="The percentage of frames that must be unmasked, excluding the frames removed by start censoring"
    )

    start_censoring = traits.Int(
        0,
        desc="Number of frames to censor out automatically at the beginning of each run."
    )


class CheckRunRetentionOutputSpec(TraitedSpec):
    valid = traits.Bool(
        default_value=False,
        desc="If this run passed the frame retention check"
    )
    retention_percentage = traits.Float(
        desc="The percent of frames retained from the tmask"
    )
    frames_retained = traits.Int(
        desc="The number of frames retained from the tmask"
    )
    total_frames = traits.Int(
        desc="The total number of frames in the tmask"
    )


class CheckRunRetention(SimpleInterface):
    input_spec = CheckRunRetentionInputSpec
    output_spec = CheckRunRetentionOutputSpec

    def _run_interface(self, runtime):

        (
            self._results["valid"], 
            self._results["retention_percentage"], 
            self._results["frames_retained"],
            self._results["total_frames"]
        ) = check_run_retention(
            tmask_file=self.inputs.tmask_file,
            retention_threshold=self.inputs.retention_threshold,
            start_censoring=self.inputs.start_censoring
        )

        return runtime


def check_run_retention(tmask_file: Path | str,
                        retention_threshold: float,
                        start_censoring: int):
    import numpy as np

    tmask_data = np.loadtxt(tmask_file)
    total_frames = tmask_data.shape[0]
    tmask_after_censor = tmask_data[start_censoring:]
    non_censored_frames = tmask_after_censor.shape[0]
    retained_frames = np.sum(tmask_after_censor)
    perc_retained = (retained_frames / non_censored_frames) * 100
    valid = perc_retained >= retention_threshold
    return (valid, perc_retained, int(retained_frames), int(total_frames))


class CheckRuntSNRInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc="Path to the bold file"
    )

    tsnr_threshold = traits.Float(
        default_value=0.0,
        desc="The lowest, whole brain tSNR value allowed"
    )

    tmask_file = traits.Union(
        File(exists=True),
        None,
        default_value=None,
        desc="Path to the tmask file"
    )

    brain_mask = traits.Union(
        traits.File(exists=True),
        None,
        default_value=None,
        desc="The brain mask that accompanies volumetric data"
    )


class CheckRuntSNROutputSpec(TraitedSpec):
    valid = traits.Bool(
        default_value=False,
        desc="If this run passed the tSNR check"
    )
    whole_brain_tsnr = traits.Float(
        desc="The average tSNR across all voxels/vertices"
    )


class CheckRuntSNR(SimpleInterface):
    input_spec = CheckRuntSNRInputSpec
    output_spec = CheckRuntSNROutputSpec

    def _run_interface(self, runtime):

        self._results["valid"], self._results["whole_brain_tsnr"] = check_run_tsnr(
            bold_file=self.inputs.bold_file,
            tsnr_threshold=self.inputs.tsnr_threshold,
            tmask_file=self.inputs.tmask_file,
            brain_mask=self.inputs.brain_mask
        )
        return runtime


def check_run_tsnr(bold_file: str | Path,
                   tsnr_threshold: float,
                   tmask_file: str | Path = None,
                   brain_mask: str = None):
    from oceanfla.utilities import load_data
    import numpy as np

    bold_data = load_data(bold_file, brain_mask)
    if tmask_file:
        tmask_data = np.loadtxt(tmask_file).astype(bool)
        bold_data = bold_data[tmask_data, :]

    tsnr_map = np.nanmean(bold_data, axis=0) / np.nanstd(bold_data, axis=0)
    whole_brain_avg_tsnr = np.nanmean(tsnr_map)
    valid = whole_brain_avg_tsnr >= tsnr_threshold
    return (valid, whole_brain_avg_tsnr)


class MakeRunExclusionTableInputSpec(BaseInterfaceInputSpec):
    run = traits.Int(
        desc="The BIDS run number"
    )
    task = traits.Str(
        desc="The BIDS task name"
    )
    start_censoring = traits.Int(
        0,
        desc="Number of frames to censor out automatically at the beginning of each run."
    )
    
    frame_retention_valid = traits.Bool(
        default_value=False,
        desc="If this run passed the frame retention check"
    )
    total_frames = traits.Int(
        desc="The total number of frames for this run"
    )
    retention_percentage = traits.Float(
        desc="The percent of frames retained from the tmask"
    )
    frames_retained = traits.Int(
        desc="The number of frames retained from the tmask"
    )

    tsnr_valid = traits.Bool(
        default_value=False,
        desc="If this run passed the tSNR check"
    )
    whole_brain_tsnr = traits.Float(
        desc="The average tSNR across all voxels/vertices"
    )
    included = traits.Bool(
        desc="If this run passed all validations and is included"
    )


class MakeRunExclusionTableOutputSpec(BaseInterfaceInputSpec):
    exclusion_table = traits.File(
        exists=True,
        desc="single row csv file containing exclusion information"
    )


class MakeRunExclusionTable(SimpleInterface):
    input_spec = MakeRunExclusionTableInputSpec
    output_spec = MakeRunExclusionTableOutputSpec

    def _run_interface(self, runtime):
        self._results["exclusion_table"] = make_exclusion_table(
            run=self.inputs.run,
            task=self.inputs.task, 
            start_censoring=self.inputs.start_censoring,
            frame_retention_valid=self.inputs.frame_retention_valid,
            total_frames=self.inputs.total_frames,
            perc_retained=self.inputs.retention_percentage,
            frames_retained=self.inputs.frames_retained,
            tsnr_valid=self.inputs.tsnr_valid,
            whole_brain_tsnr=self.inputs.whole_brain_tsnr,
            included=self.inputs.included
        )
        return runtime


def make_exclusion_table(run: int,
                         task: str,
                         start_censoring: int,
                         frame_retention_valid: bool,
                         total_frames: int,
                         perc_retained: float,
                         frames_retained: int,
                         tsnr_valid: bool,
                         whole_brain_tsnr: float,
                         included: bool):
    import pandas as pd 
    from pathlib import Path
    
    df = pd.DataFrame()
    df.loc[0, "task"] = task
    df.loc[0, "run"] = f"{int(run):02d}"
    df.loc[0, "total frames"] = int(total_frames)
    df.loc[0, "frames after start censoring"] = int(total_frames - start_censoring)
    df.loc[0, "frames retained"] = int(frames_retained)
    df.loc[0, "% frames retained"] = perc_retained
    df.loc[0, "pass frame retention check"] = frame_retention_valid
    df.loc[0, "whole brain tSNR"] = whole_brain_tsnr
    df.loc[0, "pass tSNR check"] = tsnr_valid

    df.loc[0, "included"] = included

    outfile = f"run-{run:02d}_task-{task}_desc-exclusion_table.csv"
    outfile = str(Path().resolve() / outfile)
    df.to_csv(outfile, index=False)
    return outfile
