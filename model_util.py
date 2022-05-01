
import os
from datasets import load_dataset, Dataset, DatasetDict

def find_data_folder(cur_dir):
    # successfully exit
    if os.path.exists(os.path.join(cur_dir, 'data')):
        return os.path.join(cur_dir, 'data')
    # fail exit
    par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
    if par_dir == cur_dir: # root dir
        return None
    
    # recursive call
    return find_data_folder(par_dir)







