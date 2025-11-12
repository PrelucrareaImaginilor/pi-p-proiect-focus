import os
import cv2
import numpy as np
import imageio
import argparse
from multiprocessing import Pool

T_H, T_W = 64, 64
START, FINISH, WARNING, FAIL = "START", "FINISH", "WARNING", "FAIL"

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser(description='Dataset pretreatment')
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--log_file', default='./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string)
parser.add_argument('--worker_num', default=1, type=int)
opt = parser.parse_args()

INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS = opt.input_path, opt.output_path, opt.log_file, opt.worker_num
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_print(pid, comment, msg):
    log = f"# JOB {pid} : --{comment}-- {msg}\n"
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as f:
            f.write(log)
    print(log, end='')

def cut_img(img, seq_info, frame_name, pid):
    if img is None or img.sum() <= 10000:
        log_print(pid, WARNING, f"{'-'.join(seq_info)}:{frame_name} no data")
        return None
    y = img.sum(axis=1)
    y_top, y_btm = (y != 0).argmax(), (y != 0).cumsum().argmax()
    img = img[y_top:y_btm + 1, :]
    _r = img.shape[1] / img.shape[0]
    _t_w = max(1, int(T_H * _r))
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = next((i for i, v in enumerate(sum_column) if v > sum_point / 2), -1)
    if x_center < 0:
        log_print(pid, WARNING, f"{'-'.join(seq_info)}:{frame_name} no center")
        return None
    h_T_W = T_W // 2
    left, right = x_center - h_T_W, x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        pad = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([pad, img, pad], axis=1)
        left += h_T_W
        right += h_T_W
    return img[:, left:right].astype('uint8')

def cut_pickle(seq_info, pid):
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    os.makedirs(out_dir, exist_ok=True)
    log_print(pid, START, '-'.join(seq_info))
    frame_list = sorted(f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f)))
    count = 0
    for fname in frame_list:
        img = cv2.imread(os.path.join(seq_path, fname), cv2.IMREAD_GRAYSCALE)
        img = cut_img(img, seq_info, fname, pid)
        if img is not None:
            imageio.imwrite(os.path.join(out_dir, fname), img)
            count += 1
    if count < 5:
        log_print(pid, WARNING, f"{'-'.join(seq_info)} less than 5 valid frames")
    log_print(pid, FINISH, f"{count} frames saved in {out_dir}")

if __name__ == '__main__':
    pool = Pool(WORKERS)
    id_list = sorted(os.listdir(INPUT_PATH))
    pid = 0
    results = []
    for _id in id_list:
        seq_types = sorted(os.listdir(os.path.join(INPUT_PATH, _id)))
        for st in seq_types:
            views = sorted(os.listdir(os.path.join(INPUT_PATH, _id, st)))
            for v in views:
                seq_info = [_id, st, v]
                results.append(pool.apply_async(cut_pickle, args=(seq_info, pid)))
                pid += 1
    pool.close()
    for r in results:
        r.get()
    pool.join()
