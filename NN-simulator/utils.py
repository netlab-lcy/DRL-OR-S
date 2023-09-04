import os
import glob        

def cleanup_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*'))
        for f in files:
            try:
                os.remove(f)
            except:
                pass      