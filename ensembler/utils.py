import os, errno
import functools 

def notify_when_done(fn): 
    @functools.wraps(fn)
    def print_done(*args, **kwargs):
        fn(*args, **kwargs)
        log_done()
    return print_done


def log_done():
    print('Done.')

    
def create_dir(dirname): 
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        print(f'File {dirname} exists, pass...') 
        
        
def file_exists_and_not_empty(filepath):
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 0:
            return True
    return False