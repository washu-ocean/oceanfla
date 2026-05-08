import argparse
from pathlib import Path
from textwrap import dedent
from oceanfla.oceanparse import OceanParser
from oceanfla.utilities import export_args_to_file
import logging
import bids
from datetime import datetime
from oceanfla.parser import VERSION

logger = logging.getLogger("nipype.utils")

def _build_parser():

    # Build out some useful argument types
    def ExistingPath(path):
        p = Path(path).resolve()
        if not p.exists():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing path"
            )
        return p.resolve()

    def ExistingDir(path):
        p = ExistingPath(path)
        if not p.is_dir():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing directory"
            )
        return p

    def ExistingFile(path):
        p = ExistingPath(path)
        if not p.is_file():
            raise argparse.ArgumentTypeError(
                f"path string <{path}> does not represent an existing file"
            )
        return p

    def ParentExists(path):
        p = Path(path).resolve()
        if not p.parent.exists():
            raise argparse.ArgumentTypeError(
                f"the parent directory for path string <{path}> does not exist"
            )
        return p

    def PositiveVal(val, dtype=int):
        valid = True
        out = None
        if isinstance(val, list):
            try:
                out = [dtype(v) for v in val]
                for v in out:
                    if v < 0:
                        valid = False
                        break
            except:
                valid = False
        elif isinstance(val, str):
            try:
                out = dtype(val)
                valid = (out >= 0)
            except:
                valid = False
        else:
            valid = False
        if not valid:
            raise argparse.ArgumentTypeError(
                f"The value(s) supplied must be numerical and greater than or equal to zero: {val}"
            )
        return out

    def PositiveInt(val):
        return PositiveVal(val, int)

    def AboveZeroInt(val):
        i_val = PositiveInt(val)
        if i_val <= 0:
            raise argparse.ArgumentTypeError(
                f"The value(s) supplied must be greater than zero: {val}"
            )
        return i_val

    def PositiveFloat(val):
        return PositiveVal(val, float)

    def AboveZeroFloat(val):
        f_val = PositiveFloat(val)
        if f_val <= 0:
            raise argparse.ArgumentTypeError(
                f"The value(s) supplied must be greater than zero: {val}"
            )
        return f_val

    def Percent(val):
        f_val = PositiveFloat(val)
        if f_val > 100 or f_val < 0:
            raise argparse.ArgumentTypeError(
                f"The value supplied cannot be interpreted as a percentage: {val}"
            )
        return f_val

    # Create the argument parser
    parser = OceanParser(
        prog="postfla",
        description="Ocean Labs aggregate statistics for first level analysis",
        fromfile_prefix_chars="@",
        epilog="An arguments file can be accepted with @FILEPATH"
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")

    # session_arguments = parser.add_argument_group("Session Specific")

    parser.add_argument("--fla_dir", "-i", required=True, type=ExistingDir,
                        help="The BIDS directory containing oceanfla outputs")

    parser.add_argument("--output_dir", "-o", type=Path,
                        help="Alternate Path to a directory to store the results of this analysis. Default is '[fla_dir]/postfla/'")

    parser.add_argument("--work_dir", "-w", type=ExistingDir, required=True,   
                        help="Path to a working directory to store intermediate outputs")

    parser.add_argument("--n_procs", type=PositiveInt, default=4,
                        help="The number of CPUs to use for execution")

    return parser


# Function to parse the command line arguments and
#   validate them before they become global options
def parse_args():

    parser = _build_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=False, exist_ok=True)
    else:
        args.output_dir = args.fla_dir / "postfla"





