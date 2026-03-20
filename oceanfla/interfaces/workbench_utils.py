from turtle import position
from nipype.interfaces.base import (
    TraitedSpec,
    traits,
    File,
    CommandLineInputSpec
)
from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.workbench.cifti import (
    CiftiSmooth, 
    CiftiSmoothInputSpec, 
    CiftiSmoothOutputSpec
)

from oceanfla.interfaces.utility import OptionalInterfaceSpec, OptionalCommandLineInterface


class VolumeSmoothInputSpec(OptionalInterfaceSpec, CommandLineInputSpec):
    volume_in = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input NIFTI file to smooth",
    )
    kernel = traits.Float(
        mandatory=True,
        argstr="%s",
        position=1,
        desc="the size of the gaussian smoothing kernel in mm, as sigma by default",
    )
    volume_out = File(
        hash_files=False,
        name_source=["volume_in"],
        name_template="smoothed_%s.nii",
        keep_extension=True,
        argstr="%s",
        position=2,
        desc="The output NIFTI",
    )
    roi = File(
        exists=True,
        position=3,
        argstr="-roi %s",
        desc="the NIFTI volume to use as an ROI for smoothing",
    )
    fix_zeros = traits.Bool(
        position=4,
        argstr="-fix-zeros",
        desc="treat values of zero as missing data",
    )
    subvolume = traits.Int(
        position=5,
        argstr="-subvolume %d",
        desc="the subvolume number or name to smooth"
    )

class VolumeSmoothOutputSpec(OptionalInterfaceSpec):
    volume_out = File(exists=True, desc="output NIFTI file")

class VolumeSmooth(OptionalCommandLineInterface, WBCommand):
    input_spec = VolumeSmoothInputSpec
    output_spec = VolumeSmoothOutputSpec

    _cmd = "wb_command -volume-smoothing"


class CiftiParcellateInputSpec(CommandLineInputSpec):
    cifti_in = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input CIFTI file to parcellate",
    )
    cifti_label = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="a cifti label file to use for the parcellation"
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        default="COLUMN",
        argstr="%s",
        position=2,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    cifti_out = File(
        hash_files=False,
        name_source=["cifti_in"],
        argstr="%s",
        position=3,
        desc="The parcellated CIFTI file"
    )
    method = traits.Enum(
        "MAX",
        "MIN",
        "INDEXMAX",
        "INDEXMIN",
        "SUM",
        "PRODUCT",
        "MEAN",
        "STDEV",
        "SAMPSTDEV",
        "VARIANCE",
        "TSNR",
        "COV",
        "L2NORM",
        "MEDIAN",
        "MODE",
        "COUNT_NONZERO",
        argstr="-method %s",
        position=4,
        desc="the method to use to assign parcel values from the values of member brainordinates"
    )

class CiftiParcellateOutputSpec(TraitedSpec):
    cifti_out = File(
        exists=True,
        desc="The parcellated CIFTI file"
    )

class CiftiParcellate(WBCommand):
    input_spec = CiftiParcellateInputSpec
    output_spec = CiftiParcellateOutputSpec

    _cmd = "wb_command -cifti-parcellate"

    def _overload_extension(self, value):
        value_stem, cifti_ext, nii = value.rsplit(".", 2)
        new_ext = f".p{cifti_ext[1:]}.nii"
        return value_stem + new_ext


class SurfaceSmoothInputSpec(CiftiSmoothInputSpec, OptionalInterfaceSpec):
    pass


class SurfaceSmoothOutputSpec(CiftiSmoothOutputSpec, OptionalInterfaceSpec):
    pass


class SurfaceSmooth(OptionalCommandLineInterface, CiftiSmooth):
    input_spec = SurfaceSmoothInputSpec
    output_spec = SurfaceSmoothOutputSpec
