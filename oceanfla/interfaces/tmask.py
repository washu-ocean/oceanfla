from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype import Function
from pydot import Union
from sqlalchemy import desc

class _MakeTmaskInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="Path to nuisance matrix (as a .csv or .tsv)"
    )
    fd_threshold = traits.Float(
        mandatory=True,
        desc="FD threshold for masking frames."
    )
    minimum_unmasked_neighbors = traits.Int(
        0,
        desc="""\
Number of frames to mask out on either side of each frame masked
due to motion.
"""
    )
    start_censoring = traits.Int(
        0,
        desc="Number of frames to censor out automatically at the beginning of each run."
    )
    dscans_tsv = traits.Union(
        None,
        traits.File(exists=True),
        default_value=None,
        desc="A bids style dscans file to inform the tmask censoring"
    )


class _MakeTmaskOutputSpec(TraitedSpec):
    tmask_file = File(
        exists=True,
        desc="Path to tmask (a .txt file)"
    )


class MakeTmask(SimpleInterface):
    input_spec = _MakeTmaskInputSpec
    output_spec = _MakeTmaskOutputSpec

    def _run_interface(self, runtime):
        
        self._results["tmask_file"] = make_tmask(
            confounds_file=self.inputs.confounds_file,
            fd_threshold=self.inputs.fd_threshold,
            minimum_unmasked_neighbors=self.inputs.minimum_unmasked_neighbors,
            start_censoring=self.inputs.start_censoring,
            dscans_file = self.inputs.dscans_tsv
        )

        return runtime
    

def make_tmask(confounds_file: Path | str,
               fd_threshold: int,
               minimum_unmasked_neighbors: int,
               start_censoring: int,
               dscans_file: str = None):
    from oceanfla.utilities import replace_entities
    import pandas as pd
    import numpy as np

    if start_censoring < 0:
        raise ValueError("The 'start_censoring' argument of make_tmask() must be 0 or positive.")
    if minimum_unmasked_neighbors < 0:
        raise ValueError("The 'minimum_unmasked_neighbors' argument of make_tmask() must be 0 or positive.")
    if fd_threshold < 0:
        raise ValueError("The 'fd_threshold' argument of make_tmask() must be 0 or positive.")

    df = pd.read_csv(confounds_file, sep="\t")

    fd_arr = df.loc[:, "framewise_displacement"].to_numpy()
    if minimum_unmasked_neighbors > 0:
        fd_arr_padded = np.pad(fd_arr, pad_width := minimum_unmasked_neighbors)
        fd_mask = np.full(len(fd_arr_padded), False)
        for i in range(pad_width, len(fd_arr_padded) - pad_width):
            if all(fd_arr_padded[range(i - pad_width, i + pad_width + 1)] < fd_threshold):
                fd_mask[i] = True
            elif i - pad_width < pad_width and all(fd_arr_padded[range(pad_width, i + pad_width + 1)] < fd_threshold):
                fd_mask[i] = True
            elif i + pad_width + 1 > len(fd_arr_padded) - pad_width and all(fd_arr_padded[range(i - pad_width, len(fd_arr_padded) - pad_width)] < fd_threshold):
                fd_mask[i] = True
            else:
                fd_mask[i] = False
        fd_mask = fd_mask[pad_width:-pad_width]
    else:
        fd_mask = fd_arr < fd_threshold
    fd_mask[:start_censoring] = False

    if dscans_file:
        dummy_scans_df = pd.read_csv(dscans_file, sep="\t")
        if "dummy_scan" not in dummy_scans_df.columns.to_list():
            raise RuntimeError(f"cannot find the 'dummy_scan' column in the supplied dscans file: {dscans_file}")
        dummy_scans = dummy_scans_df.loc[:, "dummy_scan"].to_numpy().astype(int)
        if len(dummy_scans) != len(fd_mask):
            raise RuntimeError(f"length of tmask: {len(fd_mask)} and dcans file: {len(dummy_scans)} are not equal")
        fd_mask[dummy_scans>0] = False

    out_file = replace_entities(
            file=confounds_file, 
            entities={
                "suffix": f"{str(fd_threshold).replace('.', 'p')}mm-tmask", 
                "ext": ".txt",
                "path": None
        })
    
    np.savetxt(out_file, fd_mask)
    return out_file


class FindDscansInputSpec(BaseInterfaceInputSpec):
    dscans_directory = traits.Directory(
        exists=True,
        mandatory=True,
        desc="Path to existing directory containing 'dscans' files (as a .csv or .tsv)"
    )
    source_file = traits.File(
        mandatory=True,
        desc="A bids file to use as a reference for bids entity values when searching the supplied directory"
    )


class FindDscansOutputSpec(TraitedSpec):
    dscans_file = traits.Union(
        File(exists=True),
        None,
        desc="Path to a dscans file (a .tsv file)"
    )


class FindDscans(SimpleInterface):
    input_spec = FindDscansInputSpec
    output_spec = FindDscansOutputSpec

    def _run_interface(self, runtime):
        
        self._results["dscans_file"] = find_dscans_file(
            dscans_dir=self.inputs.dscans_directory,
            source_bids=self.inputs.source_file
        )

        return runtime


def find_dscans_file(dscans_dir:str, 
                     source_bids:str):
    from pathlib import Path
    import pandas as pd
    from bids.layout import parse_file_entities
    from oceanfla.config import get_logger
    
    logger = get_logger("nipype.interface")

    # the entities we will use to match files
    match_entities = ["subject", "task"]
    source_entities = parse_file_entities(source_bids)
    for match_ent in match_entities:
        if match_ent not in source_entities:
            raise RuntimeError(f"Cannot find the needed entities from the source file while looking for dscans files: {match_entities}")
    match_entities.append("echo")
    match_entities.append("run")

    # search the dscans directory for possible matches
    sub, task = source_entities["subject"], source_entities["task"]
    ses = None
    if "session" in source_entities:
        ses = source_entities["session"]
        match_entities.append("session")
    dscans_files = sorted(Path(dscans_dir).glob(f"**/sub-{sub}{'_ses-'+ses if ses else ''}_task-{task}*_dscans.tsv"))

    # loop through files and see what matches all needed entities
    selected_dscans_file = None
    for dfile in dscans_files:
        dfile_entities = parse_file_entities(str(dfile.resolve()))
        all_match = True
        for match_ent in match_entities:
            if match_ent in source_entities:
                if match_ent not in dfile_entities:
                    all_match = False
                    break
                if source_entities[match_ent] != dfile_entities[match_ent]:
                    all_match = False
                    break
        if all_match:
            selected_dscans_file = str(dfile.resolve())
            break
    
    if selected_dscans_file:
        logger.info(f"found dcans file <{selected_dscans_file}> from source file <{source_bids}>")
    else:
        logger.info(f"did not find any dcans file for source file <{source_bids}>")
    return selected_dscans_file


def make_tmask_tsv(tmask_file:str, fd_threshold:float):
    import numpy as np
    import pandas as pd
    from oceanfla.utilities import replace_entities

    tmask_data = np.loadtxt(tmask_file)
    tmask_df = pd.DataFrame(columns=[f"{str(fd_threshold)}mm_tmask"], data=tmask_data)
    out_file = replace_entities(
        file=tmask_file, 
        entities={"ext": ".tsv", "path": None}
    )
    tmask_df.to_csv(out_file, sep="\t")
    return out_file