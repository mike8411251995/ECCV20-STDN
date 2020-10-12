from collections import defaultdict
from tqdm import tqdm
import os, glob
import shutil

root_dir = '/home/t-yualee/data/oulu_npu/landmarks/cropped/'
folder_list = glob.glob('/home/t-yualee/data/oulu_npu/landmarks/cropped/*/*')

liveness_dict = defaultdict(lambda: 'spoof')
liveness_dict['_1'] = 'live'

for folder in tqdm(folder_list):
    split = folder.split('/')[-2]
    liveness = liveness_dict[folder[-2:]]

    prefix = folder.split(split)[0]
    suffix = folder.split(split)[1][1:]
    dest = os.path.join(prefix, split, liveness, suffix)
    dest = shutil.move(folder, dest)
    # tqdm.write(f'prefix = {prefix}, suffix = {suffix}')
    tqdm.write(f'moving from {folder} -> {dest}') 