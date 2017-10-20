import os
from tqdm import tqdm


def map_dir(map_function, dirpath, **kwargs):
    filelist = []
    for root, _, files in os.walk(dirpath):
        for name in files:
            filelist.append(os.path.join(root, name))
    results = []
    for filename in tqdm(filelist):
        results.append(map_function(filename))

    if kwargs.get('return_list', False):
        return results, filelist
    else:
        return results
