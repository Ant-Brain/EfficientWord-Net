import glob
from os.path import isdir

def generate_reference(input_dir:str,output_dir):
    assert(isdir(input_dir))
    assert(isdir(output_dir))
