from pathlib import Path
import bids
import json
import multiprocessing
import logging
from logging.handlers import QueueHandler


class Options():
    '''
    A singleton class designed to be the holder of all user parsed arguments.
    This class can only be initialized once, afterward, the same instance is returned.
    '''
    _instance = None
    _initialized = False
    layouts = []
    logger_names = ["nipype.workflow", "nipype.utils", "nipype.interface"]
    generic_nuisance_columns = ["mean", "trend", "spike"]
    _pattern_file = Path(__file__).resolve().parent / "resources" / "bids_paths.json"
    log_format = '%(asctime)s,%(msecs)d %(name)-2s %(levelname)-2s:\n\t %(message)s'

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, opts=None):
        if not self._initialized and opts:
            # set up the global options
            option_msg_list = ["User Inputs:", "-"*30]

            for k, v in opts.items():
                # find the layouts
                if isinstance(v, bids.BIDSLayout):
                    self.layouts.append(v)
                # check for variable regroupings
                elif k == "group":
                    option_msg_list.append(f" {k} : {v}")
                    gmap = dict()
                    for regroup in v:
                        group_rename = regroup[-1]
                        for i in range(len(regroup)-1):
                            gmap[regroup[i]] = group_rename
                    setattr(self, k, gmap)
                    continue
                else:
                    option_msg_list.append(f" {k} : {v}")
                # add the option to the class attributes
                setattr(self, k, v)

            option_msg_list.append("-"*30)

            # start the logging subprocess
            log_process, log_queue = config_logging_process(self.log_file, self.log_level, self.log_format)
            setattr(self, 'log_queue', log_queue)
            setattr(self, 'log_process', log_process)
            for log_name in self.logger_names:
                logger = get_logger(log_name, log_queue)
                logger.setLevel(self.log_level)
            
            # log the arguments used for this run
            logger = get_logger("nipype.utils")
            logger.info("\n\t".join(option_msg_list))

            self._initialized = True
            self.bids_patterns = json.loads(self._pattern_file.read_text())["oceanfla_patterns"]
    

all_opts = Options()

def set_configs(args):
    all_opts.__init__(args)


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


def file_log_process(q, log_file, log_level=logging.INFO, log_fmt=None):

    # remove current handlers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if isinstance(logger, logging.Logger):
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
    # Special handling for the root logger, which is not in loggerDict but is accessible
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file)
    fmtr = logging.Formatter(log_fmt)
    file_handler.setFormatter(fmtr)
    root.addHandler(file_handler)
    root.setLevel(log_level)

    while True:
        try:
            record = q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            msg = "Error in the file logging process"
            root.error(msg)
            print(msg)
            break
    root.info("closing log file")
    file_handler.close()
    

def config_logging_process(log_file, log_level=logging.INFO, log_fmt=None):
    log_q = multiprocessing.Queue(-1)

    log_process = multiprocessing.Process(target=file_log_process, args=(log_q, log_file, log_level, log_fmt))
    log_process.start()

    logger = get_logger("nipype.utils", log_q)
    logger.info("File logging started")

    return log_process, log_q


def finish_logging():
    try: 
        logger = get_logger('nipype.utils')
        logger.info("Ending log")
    except:
        print("Ending log")
    finally:
        if all_opts.log_process:
            all_opts.log_queue.put_nowait(None)
            all_opts.log_process.join()
    logging.shutdown()
    return


def get_logger(name, q=None):
    if q is None:
        q = all_opts.log_queue
    
    logger = logging.getLogger(name)
    has_queue_hdlr = any([isinstance(hdlr, QueueHandler) 
                          for hdlr in logger.handlers])
    if not has_queue_hdlr:
        queue_handler = QueueHandler(q)
        logger.addHandler(queue_handler)

    return logger


def close_layouts():
    for lay in all_opts.layouts:
        lay.connection_manager.session.close()
    del all_opts.layouts[:]


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
