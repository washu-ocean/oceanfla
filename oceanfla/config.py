# from nipype import logging
from pathlib import Path
import bids
import json

# all_opts = None

class Options():
    '''
    A singleton class designed to be the holder of all user parsed arguments.
    This class can only be initialized once, afterward, the same instance is returned.
    '''
    _instance = None
    _initialized = False
    layouts = []
    generic_nuisance_columns = ["mean", "trend", "spike"]
    _pattern_file = Path(__file__).resolve().parent / "resources" / "bids_paths.json"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, opts=None):
        if not self._initialized and opts:
            for k, v in opts.items():
                if isinstance(v, bids.BIDSLayout):
                    self.layouts.append(v)
                setattr(self, k, v)
            global all_opts
            all_opts = self
            self._initialized = True
            self.bids_patterns = json.loads(self._pattern_file.read_text())["oceanfla_patterns"]
            # self._set_patterns()
    
    # def _set_patterns(self):
    #     default_patterns = []
    #     seen_configs = set()
    #     for l in self.layouts:
    #         for c in l.config.values():
    #             if (c.default_path_patterns is not None) and (c not in seen_configs):
    #                 default_patterns.extend(c.default_path_patterns)
    #                 seen_configs.add(c)
        
    #     extra_patterns = json.loads(self._pattern_file.read_text())["oceanfla_patterns"]
    #     self.bids_patterns = default_patterns + extra_patterns

all_opts = Options()

def set_configs(args):
    all_opts.__init__(args)
    # Options(args)
    # loggers.initialize()


def get_layout_for_file(file) -> bids.BIDSLayout:
    '''
    Function to return the corresponding bids.BIDSLayout
    for a given filepath 

    Parameters
    ----------
    file: str
        The path of the file belonging to some BIDSLayout / BIDS directory
    

    Returns
    -------
    bids.BIDSLayout
        The BIDSLayout object if the file belongs to a parsed BIDS directory
    
    '''
    if isinstance(file, Path):
        file = str(file.resolve())
    if isinstance(file, str):
        file = str(Path(file).resolve())
    else:
        raise ValueError(
            f"argument must be of type Path or str, not {type(file)}")

    for lay in all_opts.layouts:
        if file.startswith(str(lay._root.resolve())):
            return lay
    raise RuntimeError(f"No layout correspond to the input file {file}")



def get_bids_file(file:str):
    '''
    Function to return the corresponding bids.layout.BIDSFile 
    for a given filepath 

    Parameters
    ----------
    file: str
        The path of the file to convert
    

    Returns
    -------
    bids.layout.BIDSFile
        The BIDSFile object or None if it is not found
    
    '''
    file_layout = get_layout_for_file(file)
    return file_layout.get_file(file)



# class loggers():

#     _default_log_format = "%(levelname)s:%(asctime)s:%(module)s: %(message)s"
#     _log_level = logging.INFO
#     _stream_handler = logging.StreamHandler(stream=sys.stdout)

#     root = None
#     operations = None
#     utils = None

#     @classmethod
#     def initialize(cls):
#         opts = Options()
#         if opts.debug:
#             cls._log_level = logging.DEBUG

#         log_dir = opts.output_dir.parent / "logs"
#         log_dir.mkdir(parents=True, exist_ok=True)

#         log_path = log_dir / f"{opts.file_name_base}_desc-{datetime.datetime.now().strftime('%m-%d-%y_%I-%M%p')}{opts.custom_desc}.log"
#         cls._file_handler = logging.FileHandler(log_path)

#         logging.basicConfig(level=cls._log_level,
#                     handlers=[
#                         cls._stream_handler,
#                         cls._file_handler
#                     ],
#                     format=cls._default_log_format)

#         cls.root = logging.getLogger()
#         cls.operations = logging.getLogger("first_level.operations")
#         cls.utils = logging.getLogger("first_level.utils")
