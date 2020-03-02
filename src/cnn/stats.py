import argparse
from utils import dir_path, get_num_pixels
import os


def get_stats(path):
    stats = {}
    for class_ in os.listdir(path):
        if os.path.isdir(path):
            stats[class_] = {}
            for img in os.listdir(os.path.join(path, class_)):
                width, height = get_num_pixels(os.path.join(path, class_, img))
                stats[class_][img] = width, height
    return stats


def get_freqs(img_dict):
    sizes = {}
    for img in img_dict:
        size = img_dict[img]
        if size in sizes:
            sizes[size] += 1
        else:
            sizes[size] = 1
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    stats = get_stats(os.path.join('..', '..', args.path))
    res = []
    for key in stats:
        res.append(f'{key}: total = {len(stats[key])}, size_freqs = {get_freqs(stats[key])}\n\n')
    with open(os.path.join('..', '..', args.path, 'stats.txt'), 'w') as f:
        f.writelines(res)


if __name__ == '__main__':
    main()
