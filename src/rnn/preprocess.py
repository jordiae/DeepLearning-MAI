import os
import logging
import shutil
import json
from datetime import date

TMP_LOG = 'preprocess_tmp.log'
logging.basicConfig(filename=TMP_LOG, level=logging.INFO)


def write_logs(path, config):
    logging.info(f'Writing logs')
    j = json.dumps(config)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        f.write(j)
    shutil.move(TMP_LOG, os.path.join(path, 'preprocess.log'))


def preprocess(**kwargs):
    pass


if __name__ == '__main__':
    path = os.path.join('..', '..', 'data', 'rnn')  # TODO path, preprocessing
    seed = 0
    config_preprocess = dict(original_data_path='mypath', seed=seed)
    preprocess(**config_preprocess)
    write_logs(path, config_preprocess)
