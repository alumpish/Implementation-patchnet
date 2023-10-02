import os
import sys
import oulu
import rose
import msu_mfsd

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.utils import read_cfg

if __name__ == '__main__':
    cfg = read_cfg(cfg_file='config/config.yaml')

    datasets = cfg['preprocess']['datasets']

    for dataset in datasets:
        if dataset == 'rose':
            rose.main(cfg['dataset']['dirs']['rose'])
        elif dataset == 'oulu':
            oulu.main(cfg['dataset']['dirs']['oulu'])
        elif dataset == 'msu_mfsd':
            msu_mfsd.main(cfg['dataset']['dirs']['msu_mfsd'])
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
