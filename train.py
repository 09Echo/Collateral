import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import warnings
import random
from sklearn import metrics

from data_loader import ProVe
warnings.filterwarnings('ignore')
#from propose_fdiff_c_fir_prm import mpvit_small
#from repVit import repvit_m3
from unireplknet import unireplknet_t
#from MambaVision.mambavision.models.mamba_vision import mamba_vision_T
#from replknet import create_RepLKNet31B
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--b_size", type=int, default=8, help="train batch size")
    parser.add_argument("--t_size", type=int, default=10, help="train batch size")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--k_fold", type=int, default=3)
    parser.add_argument("--save_path", type=str, default='./results/unireplknet', help = 'model_save_path')
    parser.add_argument('--weights', default=[1, 2.65, 4.417], help='cross entropyloss')
    args = parser.parse_args()
    return args

RANDOM_SEED = 42 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def main(args):
    set_seed(RANDOM_SEED)

    for ki in range(args.k_fold):
        #model = repvit_m3(num_classes=3).to(args.device)
        model = unireplknet_t(num_classes=3).to(args.device)
        #model = mamba_vision_T(prtrained=True, model_path='/home/hubin/Collateral/scodeeeee/MambaVision-T-1K').cuda()
        #model = mamba_vision_T(prtrained=False).to(args.device)
        #model = create_RepLKNet31B(small_kernel_merged=False, num_classes=3).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=False)
        loss_fn = nn.CrossEntropyLoss(weight= torch.FloatTensor(args.weights).cuda()).to(args.device)

        train_ds = ProVe(flag=0, fold=ki, mode='train', n_split=args.k_fold)
        test_ds = ProVe(flag=0, fold=ki, mode='test', n_split=args.k_fold)
        train_dl = DataLoader(train_ds, batch_size=args.b_size, num_workers=8, drop_last=False, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=args.t_size, num_workers=10, drop_last=False, shuffle=False)

        # batch数量
        train_num = len(train_dl)
        test_num = len(test_dl)

        best_acc, best_f1 = 0, 0
        for epoch in range(args.epochs):

            running_loss = 0.0
            acc1 = 0.0

            model.train()
            for data, label in train_dl:
                data, label = data.to(args.device, dtype=torch.float), label.to(args.device)
                outputs = model(data)
                predict_y = torch.max(outputs, dim=1)[1]
                loss = loss_fn(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc1 += torch.eq(predict_y, label).sum().item()
            train_accurate = acc1 / len(train_ds)
            t_loss = running_loss / train_num

            test_acc = 0
            test_loss = 0.0

            t_y = []
            p_y = []
            model.eval()
            with torch.no_grad():
                for data, label, _ in test_dl:
                    data, label = data.to(args.device, dtype=torch.float), label.to(args.device)
                    t_y = t_y + label.tolist() #真实标签
                    outputs = model(data)
                    predict_y = torch.max(outputs, dim=1)[1]
                    test_acc += torch.eq(predict_y, label).sum().item()

                    p_y = p_y + predict_y.tolist()

                    batch_loss = loss_fn(outputs, label)
                    test_loss += batch_loss.item()

            test_accurate = test_acc / len(test_ds)
            v_loss = test_loss / test_num

            lr_scheduler.step()
            precision, recall, test_f1, _ = metrics.precision_recall_fscore_support(t_y, p_y, average='macro')
            cm = metrics.confusion_matrix(t_y, p_y)
            print(
                '[epoch %d] train_loss: %.3f  test_loss: %.3f  train_accuracy: %.3f test_accuracy: %.3f test_f1: %.3f' %
                (epoch + 1, t_loss, v_loss, train_accurate, test_accurate, test_f1))
            os.makedirs(args.save_path, exist_ok=True)
            OUT = True
            if test_accurate >= best_acc:
                best_acc = test_accurate
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(args.save_path, "{}fold_{}epoch_{:.2f}%acc_{:.2f}%f1.pth").format(
                        ki, epoch + 1, test_accurate * 100, test_f1))
                print("最优模型已保存.")
                print("混淆矩阵:", cm)
                OUT = False

            if test_f1 >= best_f1:
                best_f1 = test_f1
                if OUT:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(args.save_path, "{}fold_{}epoch_{:.2f}%acc_{:.2f}%f1.pth").format(
                            ki, epoch + 1, test_accurate * 100, test_f1))
                    print("最优模型已保存.")
                    print("混淆矩阵:", cm)

if __name__ == '__main__':
    args = parse_args()
    main(args)

