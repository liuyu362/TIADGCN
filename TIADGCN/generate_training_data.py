from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from scipy import interpolate

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, _ = df.shape
    data = df[:, :, 0]
    data = np.expand_dims(data, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (np.arange(num_samples) % 288) / 288.0  # 一天中的时间，范围 [0, 1)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    #dow_zero = np.zeros((num_samples, num_nodes, 1))  # 全 0 的张量
    if add_day_in_week:
        dow = (np.arange(num_samples) % 7) / 7.0  # 一周中的天数，范围 [0, 1)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
    #else:
       # feature_list.append(dow_zero)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    data = np.load(args.traffic_filename)
    df = data['data']

    # 生成训练、验证和测试数据
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.astype(np.int64).reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.astype(np.int64).reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/PEMS04test", help="Output directory.")
    parser.add_argument("--traffic_filename", type=str, default="data/PEMS04.npz", help="Raw traffic readings.")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Input sequence Length.")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Output sequence Length.")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start")
    parser.add_argument("--dow", action='store_true', help="Add day of week feature.")
    parser.add_argument("--normalizer", type=str, default='max01', help="Normalization method.")
    parser.add_argument("--column_wise", action='store_true', default='true', help="Normalize column-wise.")
    parser.add_argument("--interpolate_method", type=str, default='linear', help="Method for interpolating missing values")

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)