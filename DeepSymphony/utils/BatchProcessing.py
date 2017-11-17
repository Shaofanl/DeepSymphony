import os
from tqdm import tqdm
from multiprocessing import Pool


def map_dir(map_function, dirpath, cores=1, **kwargs):
    filelist = []
    for root, _, files in os.walk(dirpath):
        for name in files:
            filelist.append(os.path.join(root, name))

    results = []
    if cores == 1:
        for filename in tqdm(filelist):
            results.append(map_function(filename))
    else:
        pool = Pool(cores)
        results = pool.map(map_function, filelist)

    if kwargs.get('return_list', False):
        return results, filelist
    else:
        return results
