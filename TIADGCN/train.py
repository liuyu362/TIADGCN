import logging
import argparse
import time
import os
import torch
import numpy as np
import pandas as pd
import random
from torch.optim.lr_scheduler import StepLR
import util
from util import *
from model import TIADGCN
from ranger21 import Ranger
import torch.optim as optim


# 配置日志
def setup_logging(log_dir):
    """设置日志记录器，同时输出到控制台和文件"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--num_nodes", type=int, default=307, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--epochs", type=int, default=500, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--save", type=str, default="./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "-",
                    help="save path")
parser.add_argument("--es_patience", type=int, default=100, help="quit if no improvement after this many iterations")
args = parser.parse_args()


class trainer:
    def __init__(
            self,
            scaler,
            input_dim,
            channels,
            num_nodes,
            input_len,
            output_len,
            dropout,
            lrate,
            wdecay,
            device,
    ):
        self.model = TIADGCN(
            device, input_dim, channels, num_nodes, input_len, output_len, dropout
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=1)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        logger.info(f"The number of parameters: {self.model.param_num()}")
        logger.info(self.model)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  # 64 12 170 1
        output = output.transpose(1, 3)  # 64 1 170 12
        real = torch.unsqueeze(real_val, dim=1)  # 64 1 170 12
        predict = self.scaler.inverse_transform(output)  # 64 1 170 12
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    global logger  # 声明为全局变量以便trainer类使用
    logger, log_file = setup_logging(args.save + args.data + "/")
    logger.info(f"日志已保存至: {log_file}")
    logger.info(f"参数配置: {args}")

    seed_it(6666)

    data = args.data

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.channels = 128
        args.epochs = 500
        args.es_patience = 100

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.channels = 128
        args.epochs = 200
        args.es_patience = 100

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.channels = 128
        args.epochs = 400
        args.es_patience = 100

    elif args.data == "PEMS07":
        args.data = "data/PEMS07"
        args.num_nodes = 883
        args.channels = 128
        args.epochs = 500
        args.es_patience = 100

    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + str(args.channels) + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    logger.info(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
    )

    logger.info("start training...")

    for i in range(1, args.epochs + 1):
        current_lr = engine.get_current_lr()
        log_msg = f"Epoch {i}, Current Learning Rate: {current_lr:.6f}"
        print(log_msg, flush=True)
        logger.info(log_msg)

        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        #
        # noise_mean = 10  # 均值为10
        # noise_var = 500  # 方差为500
        # noise_proportion = 0.6  # 添加噪声的比例（例如，20%）
        #
        # for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
        #     # 将数据转换为numpy数组以便处理
        #     x_np = np.array(x)  # x的维度是 [64, 12, 170, 3]
        #
        #     # 随机选择要添加噪声的样本索引（例如20%的样本）
        #     num_samples = x_np.shape[0]
        #     noise_sample_indices = random.sample(range(num_samples), int(num_samples * noise_proportion))
        #
        #     # 在最后一个维度的第一个通道（索引为0的位置）添加噪声
        #     x_noisy = x_np.copy()
        #     noise = np.random.normal(noise_mean, np.sqrt(noise_var), x_noisy[noise_sample_indices, :, :, 0].shape)
        #     x_noisy[noise_sample_indices, :, :, 0] += noise
        #
        #     # 将添加噪声后的数据转换为Tensor
        #     trainx = torch.Tensor(x_noisy).to(device)  # [64, 12, 170, 3]
        #     trainx = trainx.transpose(1, 3)  # 调整维度顺序，具体操作取决于模型的输入要求
        #
        #     # 保持训练标签不变
        #     trainy = torch.Tensor(y).to(device)
        #     trainy = trainy.transpose(1, 3)
        #
        #     metrics = engine.train(trainx, trainy[:, 0, :, :])
        #     train_loss.append(metrics[0])
        #     train_mape.append(metrics[1])
        #     train_rmse.append(metrics[2])
        #     train_wmape.append(metrics[3])
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if iter % args.print_every == 0:
                log_msg = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}".format(
                    iter, train_loss[-1], train_rmse[-1], train_mape[-1], train_wmape[-1]
                )
                print(log_msg, flush=True)
                logger.info(log_msg)
        t2 = time.time()
        log_msg = "Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1))
        print(log_msg)
        logger.info(log_msg)
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()
        log_msg = "Epoch: {:03d}, Inference Time: {:.4f} secs".format(i, (s2 - s1))
        print(log_msg)
        logger.info(log_msg)
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log_msg = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}".format(
            i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape
        )
        print(log_msg, flush=True)
        logger.info(log_msg)

        log_msg = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}".format(
            i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape
        )
        print(log_msg, flush=True)
        logger.info(log_msg)

        if mvalid_loss < loss:
            log_msg = "###Update tasks appear###"
            print(log_msg)
            logger.info(log_msg)
            if i < 500:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                log_msg = "Updating! Valid Loss: {}, epoch: {}".format(mvalid_loss, i)
                print(log_msg)
                logger.info(log_msg)

        else:
            epochs_since_best_mae += 1
            log_msg = "No update"
            print(log_msg)
            logger.info(log_msg)

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    log_msg = "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time))
    print(log_msg)
    logger.info(log_msg)

    log_msg = "Average Inference Time: {:.4f} secs".format(np.mean(val_time))
    print(log_msg)
    logger.info(log_msg)

    logger.info("Training ends")
    log_msg = "The epoch of the best result：{}".format(bestid)
    print(log_msg)
    logger.info(log_msg)

    log_msg = "The valid loss of the best model: {}".format(round(his_loss[bestid - 1], 4))
    print(log_msg)
    logger.info(log_msg)

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []
    test_m = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log_msg = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
            i + 1, metrics[0], metrics[2], metrics[1], metrics[3]
        )
        print(log_msg)
        logger.info(log_msg)

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log_msg = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
    )
    print(log_msg)
    logger.info(log_msg)

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")