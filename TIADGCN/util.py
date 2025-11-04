import numpy as np
import os
import scipy.sparse as sp
import torch


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    # Data format
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    # 对顺序出现的数据全局随机打乱
    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    data["scaler"] = scaler

    return data
    
# def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
#     data = {}
#     for category in ["train", "val", "test"]:
#         # 构建文件路径
#         file_path = os.path.join(dataset_dir, category + ".npz")
#         # 检查文件是否存在
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"数据集文件不存在：{file_path}")
#         # 加载数据
#         cat_data = np.load(file_path)
#         # 检查文件中是否有"x"和"y"键
#         if "x" not in cat_data or "y" not in cat_data:
#             raise KeyError(f"文件 {file_path} 中缺少 'x' 或 'y' 键")
#         # 赋值到data字典
#         data["x_" + category] = cat_data["x"]
#         data["y_" + category] = cat_data["y"]
#         print(f"成功加载 {category} 数据，x形状：{data['x_' + category].shape}")  # 打印数据形状
#
#     # ================ 噪声添加逻辑（此时data中一定有"x_train"） ================
#     # 噪声参数
#     noise_mean = 10.0
#     noise_var = 500.0
#     noise_std = np.sqrt(noise_var)
#     noise_prob = 0.6  # 20%样本添加噪声
#
#     # 获取训练集第一个特征（原始数据）
#     train_feat = data["x_train"][..., 0].copy()  # 用copy()确保修改生效
#     num_samples = train_feat.shape[0]
#
#     # 打印添加噪声前的统计量
#     print(f"噪声添加前 - 训练集特征均值: {train_feat.mean():.2f}, 标准差: {train_feat.std():.2f}")
#
#     # 生成噪声掩码（确保有样本被选中）
#     noise_mask = np.random.choice([True, False], size=num_samples, p=[noise_prob, 1 - noise_prob])
#     print(f"选中添加噪声的样本数: {noise_mask.sum()} (总样本数: {num_samples})")
#
#     # 添加噪声
#     if noise_mask.any():
#         noise = np.random.normal(
#             loc=noise_mean,
#             scale=noise_std,
#             size=train_feat[noise_mask].shape
#         )
#         train_feat[noise_mask] += noise
#         data["x_train"][..., 0] = train_feat  # 更新回data字典
#
#         # 打印添加噪声后的统计量
#         print(
#             f"噪声添加后 - 训练集特征均值: {data['x_train'][..., 0].mean():.2f}, 标准差: {data['x_train'][..., 0].std():.2f}")
#     else:
#         print("警告：未选中任何样本添加噪声，尝试提高noise_prob（如0.5）")
#     # ======================================================================
#
#     # 标准化逻辑（基于加噪声后的数据）
#     scaler = StandardScaler(
#         mean=data["x_train"][..., 0].mean(),
#         std=data["x_train"][..., 0].std()
#     )
#     print(f"标准化参数 - 均值: {scaler.mean:.2f}, 标准差: {scaler.std:.2f}")
#
#     # 对所有数据集的第一个特征进行标准化
#     for category in ["train", "val", "test"]:
#         data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
#
#     # 数据打乱（保持原逻辑）
#     print("Perform shuffle on the dataset")
#     random_train = torch.arange(int(data["x_train"].shape[0]))
#     random_train = torch.randperm(random_train.size(0))
#     data["x_train"] = data["x_train"][random_train, ...]
#     data["y_train"] = data["y_train"][random_train, ...]
#
#     random_val = torch.arange(int(data["x_val"].shape[0]))
#     random_val = torch.randperm(random_val.size(0))
#     data["x_val"] = data["x_val"][random_val, ...]
#     data["y_val"] = data["y_val"][random_val, ...]
#
#     # 构建DataLoader
#     data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
#     data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size or batch_size)
#     data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size or batch_size)
#     data["scaler"] = scaler
#
#     return data

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss


def metric(pred, real):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape
