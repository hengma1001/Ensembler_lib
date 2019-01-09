import os, errno, logging
import functools 
from .core import logger, mpistate


def mpirank0only(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if mpistate.rank == 0:
            fn(*args, **kwargs)
    return wrapper


def mpirank0only_and_end_with_barrier(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if mpistate.rank == 0:
            fn(*args, **kwargs)
        mpistate.comm.Barrier()
    return wrapper

def notify_when_done(fn): 
    @functools.wraps(fn)
    def print_done(*args, **kwargs):
        fn(*args, **kwargs)
        log_done()
    return print_done


def log_done():
    logger.info('Done.')

    
def create_dir(dirname): 
    try:
        os.mkdir(dirname) 
        logger.debug('Created directory {}, continuing...'.format(dirname))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        logger.debug('Directory {} exists, pass...'.format(dirname)) 
        
        
def file_exists_and_not_empty(filepath):
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 0:
            return True
    return False

def set_loglevel(loglevel):
    """
    Set minimum level for logging
    >>> set_loglevel('info')   # log all messages except debugging messages. This is generally the default.
    >>> set_loglevel('debug')   # log all messages, including debugging messages

    Parameters
    ----------
    loglevel: str
        {debug|info|warning|error|critical}
    """
    if loglevel is not None:
        loglevel_obj = getattr(logging, loglevel.upper())
        logger.setLevel(loglevel_obj)

