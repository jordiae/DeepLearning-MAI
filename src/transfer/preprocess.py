import os
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import logging
import shutil
import json
from transfer.stats import get_stats_freqs
from datetime import date
from PIL import UnidentifiedImageError
import numpy as np

TMP_LOG = 'preprocess_tmp.log'
logging.basicConfig(filename=TMP_LOG, level=logging.INFO)


def read_data(path, remove_bw=True):
    logging.info(f'Reading original data from {path}')
    if remove_bw:
        logging.info('BW images will be removed')
    dataset = {}
    mal = 0
    rgba = 0
    p = 0
    bw = 0
    for class_ in os.listdir(path):
        if os.path.isdir(os.path.join(path, class_)):
            dataset[class_] = []
            for img_path in os.listdir(os.path.join(path, class_)):
                try:
                    img = Image.open(os.path.join(path, class_, img_path))
                except UnidentifiedImageError as e:
                    mal += 1
                    logging.warning(e)
                    logging.warning(f'Detected malformatted image: {img_path}. Skipping.')
                    continue
                if img.mode == 'RGBA':
                    rgba += 1
                    logging.warning(f'Detected RGBA: {img_path}. Converting into RGB.')
                    img = png2jpg(img)
                    img.save(os.path.join(path, class_, img_path))
                if img.mode == 'P':
                    p += 1
                    logging.warning(f'Detected P: {img_path}. Converting into RGB.')
                    img = img.convert('RGB')
                    img.save(os.path.join(path, class_, img_path))
                if img.mode == 'L':
                    bw += 1
                    logging.warning(f'Detected BW image: {img_path}')
                    if remove_bw:
                        logging.warning('Skipping.')
                        continue
                dataset[class_].append(img_path)
    logging.info(f'{mal} malformatted images')
    logging.info(f'{rgba} RGBA images')
    logging.info(f'{p} P images')
    logging.info(f'{bw} BW images')
    return dataset


def build_dataset(data):
    X = []
    y = []
    for class_ in data:
        for img_path in data[class_]:
            X.append(img_path)
            y.append(class_)
    return np.array(X), np.array(y)


def stratified_split(data, orig_path, stratified_path, split_proportions=(0.8, 0.1, 0.1), seed=0):
    X, y = build_dataset(data)
    logging.info(f'Splitting data into train, valid, test')
    sss = StratifiedShuffleSplit(test_size=split_proportions[2], random_state=seed, n_splits=1)
    spl_test = list(sss.split(X, y))
    learn_indices, test_indices = spl_test[0][0], spl_test[0][1]
    sss = StratifiedShuffleSplit(test_size=split_proportions[1]/(split_proportions[0]+split_proportions[1]),
                                 random_state=seed, n_splits=1)
    X_learn = X[learn_indices]
    y_learn = y[learn_indices]
    spl_valid = list(sss.split(X[learn_indices], y[learn_indices]))
    train_indices, valid_indices = spl_valid[0][0], spl_valid[0][1]
    os.mkdir(os.path.join(stratified_path, 'train'))
    for idx in train_indices:
        img_path, class_ = X_learn[idx], y_learn[idx]
        if not os.path.exists(os.path.join(stratified_path, 'train', class_)):
            os.makedirs(os.path.join(stratified_path, 'train', class_))
        shutil.move(os.path.join(orig_path, class_, img_path), os.path.join(stratified_path, 'train', class_,
                                                                            os.path.basename(img_path)))
    os.mkdir(os.path.join(stratified_path, 'valid'))
    for idx in valid_indices:
        img_path, class_ = X_learn[idx], y_learn[idx]
        if not os.path.exists(os.path.join(stratified_path, 'valid', class_)):
            os.makedirs(os.path.join(stratified_path, 'valid', class_))
        shutil.move(os.path.join(orig_path, class_, img_path), os.path.join(stratified_path, 'valid', class_,
                                                                            os.path.basename(img_path)))
    os.mkdir(os.path.join(stratified_path, 'test'))
    for idx in test_indices:
        img_path, class_ = X[idx], y[idx]
        if not os.path.exists(os.path.join(stratified_path, 'test', class_)):
            os.makedirs(os.path.join(stratified_path, 'test', class_))
        shutil.move(os.path.join(orig_path, class_, img_path), os.path.join(stratified_path, 'test', class_,
                                                                            os.path.basename(img_path)))


def png2jpg(png):
    # Credits: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    png.load()
    background = Image.new('RGB', png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    return background


def resize(original_data, original_data_path, new_size, resized_path):
    logging.info(f'Resizing images')
    for class_ in original_data:
        if not os.path.exists(os.path.join(resized_path, class_)):
            os.makedirs(os.path.join(resized_path, class_))
        for img_path in original_data[class_]:
            img_path = os.path.join(original_data_path, class_, img_path)
            img = Image.open(img_path)
            img = img.resize(new_size, Image.LANCZOS)  # PIL.Image.LANCZOS (a high-quality downsampling filter
            img.save(os.path.join(resized_path, os.path.join(*img_path.split('/')[-2:])))


def write_logs(path, config):
    logging.info(f'Writing logs')
    j = json.dumps(config)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        f.write(j)
    shutil.move(TMP_LOG, os.path.join(path, 'preprocess.log'))


def preprocess(original_data_path, remove_bw=True, resize_data=False, new_size=None, resized_path=None, split=True,
               stratified_path=None, split_proportions=(0.8, 0.1, 0.1), seed=0):
    original_data = read_data(original_data_path, remove_bw)
    s = get_stats_freqs(original_data, original_data_path)
    for line in s:
        logging.info(line)
    data = original_data
    from_path = original_data_path
    if resize_data:
        resize(original_data, original_data_path, new_size, resized_path)
        #resized_data = read_data(original_data)
        #data = resized_data
        from_path = resized_path
    if split:
        stratified_split(data, from_path, stratified_path, split_proportions, seed)
        if resize_data:
            shutil.rmtree(resized_path)


if __name__ == '__main__':
    path = os.path.join('..', '..', 'data', 'mit67', 'not-resized')
    logging.info(date.today())
    original_data_path = os.path.join('..', '..', 'data', 'mit67', 'Images')
    remove_bw = True
    resize_data = True
    new_size = (256, 256)
    resized_path = os.path.join('..', '..', 'data', 'mit67', f'{new_size[0]}x{new_size[1]}')
    if resize_data:
        if not os.path.exists(resized_path):
            os.mkdir(resized_path)
        path = resized_path
    split = True
    stratified_path = None
    split_proportions = (0.8, 0.1, 0.1)
    seed = 0
    if split:
        stratified_path = path + '-split'
        if not os.path.exists(stratified_path):
            os.mkdir(stratified_path)
        path = stratified_path
    config_preprocess = dict(original_data_path=original_data_path, remove_bw=remove_bw, resize_data=resize_data,
                             new_size=new_size, resized_path=resized_path, split=split, stratified_path=stratified_path,
                             split_proportions=split_proportions, seed=seed)
    preprocess(**config_preprocess)
    write_logs(path, config_preprocess)
