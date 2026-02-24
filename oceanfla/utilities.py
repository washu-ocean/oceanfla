from pathlib import Path
import shutil
import nibabel as nib
import nilearn.masking as nmask
from nilearn.image import resample_img
import numpy as np
import bids
import os
import json
import logging
import sys
import re
from collections.abc import Iterable
from oceanfla.config import finish_logging

from niworkflows.utils.spaces import NONSTANDARD_REFERENCES
from templateflow import api as tflow

logger = logging.getLogger("nipype.utils")

cifti_files = [
    ".dtseries.nii",
    ".ptseries.nii",
    ".dscalar.nii",
    ".pscalar.nii",
]


def find_subjects(layout:bids.BIDSLayout):
    pass


# Grab BOLD files from the preprocessed outputs, and list the runs and file extension for each 'func_space' in the files.

def parse_session_bold_files(layout:bids.BIDSLayout, subject:str, session:str, tasks:list[str], brain_mask=None):
    '''
    Finds all BOLD files in the given BIDSLayout, with the given filters, and 
    returns a dictionary with organization {SPACE : {"runs" : {TASK : list[RUN#]}, "extension" : [FILE-EXTENSION] }} 

    Parameters
    ----------
    layout: bids.BIDSLayout
        The BIDS layout to query

    subject: str
        The BIDS subject ID to filter on

    session: str
        The BIDS session ID to filter on

    tasks: list[str]
        A list of BIDS task IDs to filter on


    Returns
    -------
    dict[str : dict[str : list] | list]
        A dictionary organizing BOLD run numbers by task and functional space

    '''
    files = layout.get(subject=subject, session=session, task=tasks, suffix="bold", datatype="func", extension=[".nii",".nii.gz",".dtseries.nii"])
    space_run_dict = dict()
    for f in files:
        # get entities
        # run = f.entities["run"] if "run" in f.entities else PaddedInt('01')
        space = f.entities["space"] if "space" in f.entities else "func"
        task = f.entities["task"]
        if space in space_run_dict:
            if task in space_run_dict[space]:
                space_run_dict[space][task].append(f.path)
            else:
                space_run_dict[space][task] = [f.path]
        else:
            space_run_dict[space] = {task:[f.path]}
    return space_run_dict


def load_data(func_file: str | Path | bids.layout.BIDSFile,
              brain_mask: str | Path | None) -> np.ndarray:
    '''
    Loads NIFTI or CIFTI file data into an in-memory numpy array of
    of shape [volume/time, voxel/vertex #]. Volumetric NIFTI files must also have
    an accompanying brain mask to define where to grab the data from.

    Parameters
    ----------
    func_file: str|Path|bids.layout.BIDSFile
        The path to a BOLD NIFTI or CIFTI file

    brain_mask: str|Path
        The path to a volumetric brain mask

    Returns
    -------
    numpy.ndarray
        A numpy array representing the data in the input file

    '''
    if isinstance(func_file, bids.layout.BIDSFile):
        func_file = str(func_file.path)
    elif not isinstance(func_file, str):
        func_file = str(func_file)
    func_img = nib.load(func_file)
    if is_cifti_file(func_file):
        return func_img.get_fdata()
    elif is_nifti_file(func_file):
        if brain_mask is not None:
            return nmask.apply_mask(func_img, brain_mask)
        else:
            func_space = re.search(r'space-([a-zA-Z0-9]+)', func_file).group(1)
            if func_space in NONSTANDARD_REFERENCES:
                raise RuntimeError(f'Volumetric data in nonstandard space {func_space} '
                                   'must use an accompanying brain mask.')
            func_res = func_img.header.get("pixdim")[1:4]
            mask_paths = tflow.get(
                func_space,
                desc="brain",
                suffix="mask"
            )
            if len(mask_paths) == 0:
                raise RuntimeError(f"Could not find a TemplateFlow brain mask for {func_space}.")
            closest_res, closest_res_path = np.inf, None
            for mask_path in mask_paths:
                mask_res = nib.load(mask_path).header.get("pixdim")[1:4]
                if np.sum(func_res - mask_res) < closest_res:
                    closest_res = np.sum(func_res - mask_res)
                    closest_res_path = mask_path
            matched_res_mask = resample_img(
                nib.load(closest_res_path),
                target_affine=func_img.affine,
                interpolation="nearest"
            )
            return nmask.apply_mask(func_img, matched_res_mask)
            # raise RuntimeError("Volumetric data must also have an accompanying brain mask")


def is_cifti_file(file: str | Path) -> str | None:
    '''
    Returns the CIFTI extention if the input path describes
    a CIFTI file, else None

    Parameters
    ----------
    file: str|Path
        The path to a BOLD NIFTI or CIFTI file

    Returns
    -------
    str | None
        The CIFTI extention or None
    '''
    if isinstance(file, Path):
        file = str(file)
    suffix = [cf for cf in cifti_files if file.endswith(cf)]
    return suffix[0] if len(suffix) > 0 else None


def is_nifti_file(file: str | Path) -> str | None:
    '''
    Returns the NIFI extention if the input path describes
    a NIFTI file, else None

    Parameters
    ----------
    file: str|Path
        The path to a BOLD NIFTI or CIFTI file

    Returns
    -------
    str | None
        The NIFTI extention or None
    '''
    if isinstance(file, Path):
        file = str(file)
    not_cifti = is_cifti_file(file) is None
    suffix = [nf for nf in [".nii.gz", ".nii"] if file.endswith(nf)]
    return suffix[0] if (len(suffix) > 0) and not_cifti else None


def create_image_like(data: np.ndarray,
                      out_file: str | Path,
                      source_header=None,
                      scalar_axis:list[str] = None,
                      brain_mask: str = None):
    """
    Create a NIFTI or CIFTI image file from data using a source header template.

    Parameters
    ----------
    data : np.ndarray
        A numpy array representing the image data to be saved.
    source_header : str | Path | nibabel.cifti2.cifti2.Cifti2Header
        The path to a CIFTI file or a Cifti2Header object to use as a template
        for the output image structure.
    out_file : str | Path
        The output file path where the NIFTI or CIFTI image will be saved.
    scalar_axis : list[str], optional
        A list of scalar axis names. If provided, creates a ScalarAxis with these names.
        If not provided, creates a SeriesAxis based on the source header. Default is None.
    brain_mask : str, optional
        Path to a volumetric brain mask file. If provided, the data is unmasked using this mask
        and saved directly into a NIFTI file. Default is None.

    Returns
    -------
    None
        The function saves the image to disk and returns None.

    Raises
    ------
    ValueError
        If source_header is not one of the following types: str, pathlib.Path, or
        nibabel.cifti2.cifti2.Cifti2Header.

    Notes
    -----
    - If brain_mask is provided, the function bypasses CIFTI image creation and
        uses nibabel's unmask function to create a volumetric NIFTI.
    - The output CIFTI image uses either a ScalarAxis or SeriesAxis for the first
        dimension, depending on whether scalar_axis is provided.
    """

    data_img = None
    if source_header:
        wrong_type = True
        if isinstance(source_header, str) or isinstance(source_header, Path):
            source_header = nib.load(source_header).header
            wrong_type = False
        if isinstance(source_header, nib.cifti2.cifti2.Cifti2Header):
            wrong_type = False
            step_size = getattr(source_header.get_axis(0), "step") if hasattr(source_header.get_axis(0), "step") else 1
            ax0 = (
                nib.cifti2.cifti2_axes.ScalarAxis(name=scalar_axis)
            ) if scalar_axis else (
                nib.cifti2.cifti2_axes.SeriesAxis(start=0, step=step_size, size=data.shape[0])
            )

            data_img = nib.cifti2.cifti2.Cifti2Image(data, (ax0, source_header.get_axis(1)))
        if wrong_type:
            raise ValueError("source_header must be one of the following types: [str, pathlib.Path, nibabel.cifti2.cifti2.Cifti2Header]")

    if brain_mask and data_img is None:
        data_img = nmask.unmask(data, brain_mask)

    if data_img is None:
        raise ValueError("Must supply either the brain_mask argument (for NIFTI) or an accepted source_header argument (for CIFTI), but neither were found")

    nib.save(data_img, out_file)
    return


def replace_entities(file:str, entities:dict):
    """
    This function iterates through a dictionary of entities and replaces each
    entity in the BIDS filename with its associated value by calling replace_entity
    for each pair.

    Parameters
    ----------
    file : str
        The file path in which entities will be replaced.
    entities : dict
        A dictionary where keys are entity names and values are the replacements
        for those entities.

    Returns
    -------
    str
        The file path with all entities replaced by their corresponding values.

    Examples
    --------
    >>> file_path = "sub-01_task-rest_bold.nii.gz"
    >>> entities = {"sub": "02", "task": "motor"}
    >>> replace_entities(file_path, entities)
    "sub-02_task-motor_bold.nii.gz"
    """

    for entity, value in entities.items():
        file = replace_entity(file, entity, value)
    return file


def replace_entity(file: str, entity: str, value: str) -> str:
    """
    Replace or modify a specific entity within a filename.

    Parameters
    ----------
    file : str
        The filename to be modified.
    entity : str
        The BIDS entity key to replace.
    value : str
        The new value for the entity. If None, 
        the entity will be removed from the filename.

    Returns
    -------
    str: 
        The modified filename with the specified entity replaced.
        If the entity is not found in the filename, returns the original filename unchanged.

    Examples
    --------
    >>> replace_entity("prefix_suffix.txt", "suffix", "newsuffix")
    "prefix_newsuffix.txt"
    >>> replace_entity("file.txt", "ext", ".md")
    "file.md"
    >>> replace_entity("file.txt", "path", "/new/path")
    "/new/path/file.txt"
    >>> replace_entity("prefix_type-value_suffix.txt", "type", "newvalue")
    "prefix_type-newvalue_suffix.txt"
    """

    if entity == "suffix":
        prefix, suffix = file.rsplit("_", 1)
        ext = suffix.split(".",1)[-1]
        return f"{prefix}_{value}.{ext}"

    if entity == "ext":
        return f"{file.split('.',1)[0]}{value}"

    if entity == "path":
        fname = Path(file).name
        if not value:
            return str(Path().resolve() / fname)
        else:
            return f"{value}/{fname}"

    entity_label = f"_{entity}-"
    if entity_label in file:
        prefix, suffix = file.split(entity_label, 1)
        suffix = suffix.split("_",1)[-1]
        if value is None:
            return f"{prefix}_{suffix}"
        return f"{prefix}{entity_label}{value}_{suffix}"

    return file


def make_option(value,
                key: str = None,
                delimeter: str = " ",
                convert_underscore: bool = False):
    """
    Generate a string, representing an option that gets fed into a subprocess or script.

    Parameters
    ----------
    value: any 
        Value to pass in along with the 'key' param.
    key: str
        Name of option, without any hyphen at the beginning.
    delimeter: str
        character to separate the key and the value in the option string. Default is a space.
    convert_underscore: bool
        flag to indicate that underscores should be replaced with '-'

    Returns
    -------
    str
        String to pass as an option into a subprocess call
    """
    second_part = None
    if key and convert_underscore:
        key = key.replace("_", "-")
    if not value:
        return ""
    elif type(value) == bool and value:
        second_part = " "
    elif type(value) == list:
        second_part = f"{delimeter}{' '.join([str(v) for v in value])}"
    else:
        second_part = f"{delimeter}{str(value)}"
    return f"--{key}{second_part}" if key else second_part


def export_args_to_file(args,
                        argument_group,
                        file_path: Path,
                        extra_args:dict = None):
    """
    Takes the arguments in the argument group, and exports their names and values in the 'args'
    namespace to a file specified at 'file_path'. The input 'file_path' can either be a txt
    file or a json file.

    Parameters
    ----------
    args: argparse.Namespace
        an argument namespace to pull input values from
    argument_group: argparse._ArgumentGroup
        The argument group representing the subset of inputs to save to a file
    file_path: pathlib.Path
        a path to a file where the arguments should be saved
    extra_args: dict
        A dictionary representing addtional argumnets to save out not present in the args Namespace
    """

    all_opts = dict(args._get_kwargs())
    opts_to_save = dict()
    for a in argument_group._group_actions:
        if a.dest in all_opts and all_opts[a.dest]:
            if type(all_opts[a.dest]) == bool:
                opts_to_save[a.option_strings[0]] = ""
            elif isinstance(all_opts[a.dest], Path):
                opts_to_save[a.option_strings[0]] = str(all_opts[a.dest].resolve())
            elif isinstance(all_opts[a.dest], Iterable) and not isinstance(all_opts[a.dest], str):
                opts_to_save[a.option_strings[0]] = [v if isinstance(v, Iterable) else str(v) for v in all_opts[a.dest]]
            else:
                opts_to_save[a.option_strings[0]] = all_opts[a.dest]

    if extra_args:
        for key, val in extra_args.items():
            opt_key = make_option(True, key).strip()
            if val:
                if isinstance(val, bool):
                    opts_to_save[opt_key] = ""
                elif isinstance(val, Path):
                    opts_to_save[opt_key] = str(val.resolve())
                else:
                    opts_to_save[opt_key] = val

    with open(file_path, "w") as f:
        if file_path.suffix == ".json":
            f.write(json.dumps(opts_to_save, indent=4))
        else:
            for k,v in opts_to_save.items():
                f.write(f"{k}{make_option(value=v)}\n")


def prompt_user_continue(msg:str) -> bool:
    """
    Prompt the user to continue with a custom message.

    :param msg: prompt message to display.
    :type msg: str

    """
    prompt_msg = f"{msg} \n\t---(press 'y' for yes, other input will mean no)"
    user_continue = input(prompt_msg + "\n")
    ans = (user_continue.lower() == "y")
    logger.debug(f"User Prompt: {prompt_msg}")
    logger.debug(f"User Response:  {user_continue} ({ans})")
    return ans


def clean_paths(path_list):
    all_good = True
    for p in path_list:
        path = Path(p)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                os.remove(path)
            logger.info(f"removed the path: {path}")
        except:
            if path.exists():
                all_good = False
                logger.error(f"ERROR removing the path: {path}")
    return all_good


def logger_exception_hook(exctype, value, traceback):
    sys.__excepthook__(exctype, value, traceback)
    try:
        logger.critical(f'Uncaught exception: {exctype.__name__} - {value}')
        while traceback:
            filename = traceback.tb_frame.f_code.co_filename
            name = traceback.tb_frame.f_code.co_name
            line_no = traceback.tb_lineno
            traceback = traceback.tb_next
            if traceback:
                logger.critical(f"-- File {filename} line {line_no}, in {name} ")

        # Where the exception occured
        logger.exception(f"File {filename} line {line_no}, in {name}", exc_info=(exctype, value, traceback))
    except:
        print("An unexpected error occured with the logging :(")
    finally:
        finish_logging()


sys.excepthook = logger_exception_hook
