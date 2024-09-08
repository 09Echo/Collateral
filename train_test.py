import math
import sys
import os
import pandas as pd
from medpy.metric import specificity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch import optim, nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pingouin as pg
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from scipy.stats import pearsonr
import warnings
from torchmetrics import Specificity

from mip_mca_selected import ProVe
warnings.filterwarnings('ignore')
# from mvit.config.defaults import get_cfg
# from mvit.models.mvit_model import MViT
# from pyramid_tnt import ptnt_s_patch16_256
# from resnet2d import ResNet18
# from ViTAE.vitaev2 import ViTAEv2_S, ViTAEv2_48M
# from swin_transformer import SwinTransformer
# from replknet import create_RepLKNet31B
# from maxvit import max_vit_small_224
# from fastnet import create_FasterNet
# from RepViT import repvit_m1
# from medvit import MedViT_small
# from convnextv2 import convnextv2_tiny
# from convnext import convnext_tiny
# from coatnet import coatnet_1
# from resnet2d import ResNet34
# from reprodoce_conv import ConvNet
from propose_fdiff_c_fir_prm import mpvit_small



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 8
lr = 3e-4
epochs = 150
device = torch.device('cuda')
save_path = '/home/hubin/Collateral/Result/eeee'

def main(train_loader, val_loader, fold, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0,
                          nesterov=True,
                          )
    lr_scheduler = CosineAnnealingLR(optimizer, epochs, 1e-5, verbose=False)
    # criterion = FocalLoss(to_onehot_y=True, weight=[0.2, 0.3, 0.5])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.75, 1.1262, 1.29])).float(),
                                    size_average=True).to(device)
    best_acc, best_epoch = 0, 0
    global_step = 0
    for epoch in range(epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = []
        val_loss = 0.0
        epoch_start_time = time.time()

        model.train()
        for data, label in train_loader:
            data, label = data.to(device, dtype=torch.float), label.to(device)
            output = model(data)
            # output = model(data, data)
            batch_loss = criterion(output, label)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().data.numpy())
            train_loss += batch_loss.item()
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for i, (data, label, _) in enumerate(val_loader):
                data, label = data.to(device, dtype=torch.float), label.to(device)
                val_pred = model(data)
                # val_pred = model(data, data)
                batch_loss = criterion(val_pred, label)

                val_acc.append(np.mean(np.argmax(val_pred.cpu().data.numpy(), axis=1) == label.cpu().data.numpy()))
                val_loss += batch_loss.item()
            v_acc = np.mean(val_acc).item()
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                  (epoch + 1, epochs, time.time() - epoch_start_time,
                   train_acc / len(train_set), train_loss / train_set.__len__(), np.mean(val_acc).item(),
                   val_loss / val_set.__len__()))
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, f'fold_{fold}_model.pt'))
            if v_acc >= best_acc:
                best_acc = v_acc
                print('Best Acc: ', best_acc)
                if best_acc >= 0.84:
                    torch.save(model.state_dict(), os.path.join(save_path, f'best_fold_{fold}_model.pt'))


def evaluate1(val_loader, fold, model):
    model = model
    model.eval()
    pred_list = []
    prob_list = []
    label_list = []
    id_list = []
    for img, label, id in val_loader:
        img = img.to(device, dtype=torch.float)
        output = model(img)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        prob = prob.cpu().data.numpy()
        pred = pred.cpu().data.numpy()
        id_list.extend([id[i].item() for i in range(len(img))])
        pred_list.extend([pred[i].item() for i in range(len(img))])
        prob_list.append(prob)
        label_list.extend([label[i].item() for i in range(len(img))])
    probability = np.vstack(prob_list)
    result = np.array([pred_list, label_list, id_list]).T
    result = pd.DataFrame(result, columns=['pred', 'label', 'id'])
    result_path = os.path.join(save_path, f'fold{fold}')
    os.makedirs(result_path, exist_ok=True)
    result.to_csv(os.path.join(result_path, 'id_result.csv'), index=False)
    np.save(os.path.join(result_path, 'probability'), probability)
    return pred_list, probability, label_list


def evaluate2(val_loader, fold, model):
    model = model
    model.eval()
    pred_list = []
    prob_list = []
    label_list = []
    id_list = []
    for img, label, id in val_loader:
        img = img.to(device, dtype=torch.float)
        output = model(img)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        prob = prob.cpu().data.numpy()
        pred = pred.cpu().data.numpy()
        id_list.extend([id[i].item() for i in range(len(img))])
        pred_list.extend([pred[i].item() for i in range(len(img))])
        prob_list.append(prob)
        label_list.extend([label[i].item() for i in range(len(img))])
    probability = np.vstack(prob_list)
    result = np.array([pred_list, label_list, id_list]).T
    result = pd.DataFrame(result, columns=['pred', 'label', 'id'])
    result_path = os.path.join(save_path, f'best_fold{fold}')
    os.makedirs(result_path, exist_ok=True)
    result.to_csv(os.path.join(result_path, 'best_id_result.csv'), index=False)
    np.save(os.path.join(result_path, 'best_probability'), probability)
    return pred_list, probability, label_list


def get_data(dataframe):
    dataframe['Judge1'] = 'pred'
    dataframe['Judge2'] = 'label'
    dataframe['ID'] = dataframe['id']
    columns = ['pat', 'Judge', 'Score']
    n = 2 * len(dataframe)
    result = pd.DataFrame({'ID': np.empty(n, dtype=str), 'Judge': np.empty(n, dtype=str),
                           'Score': np.empty(n, dtype=int)})
    result['ID'][0::2] = dataframe['ID']
    result['ID'][1::2] = dataframe['ID']
    result['Judge'][0::2] = dataframe['Judge1']
    result['Judge'][1::2] = dataframe['Judge2']
    result['Score'][0::2] = dataframe['pred']
    result['Score'][1::2] = dataframe['label']
    result['Score'] = result['Score'].astype('int')
    return result


def score(cata='', fold=2):
    Speci = Specificity(num_classes=3, threshold=1. / 5, average="macro", task="multiclass")
    df = pd.DataFrame(columns=['pred', 'label', 'id'])
    prob = []
    for i in range(fold):
        path = os.path.join(save_path, f'{cata}fold{i}')
        result_path = os.path.join(path, f'{cata}id_result.csv')
        prob_path = os.path.join(path, f'{cata}probability.npy')
        data1 = pd.read_csv(result_path)
        prob1 = np.load(prob_path)
        df = [df, data1]
        df = pd.concat(df)
        prob.extend([prob1[i] for i in range(len(prob1))])
    prob = np.array(prob)

    df = df[['pred', 'label']]
    df['id'] = df.index

    pred = df['pred'].tolist()
    label = df['label'].tolist()
    prob1 = prob[:, 2]
    prob2 = 1 - prob[:, 0]
    result = get_data(df)
    icc = pg.intraclass_corr(data=result, targets='ID', raters='Judge', ratings='Score')
    kappa1 = cohen_kappa_score(pred, label)
    kappa2 = cohen_kappa_score(pred, label, weights='quadratic')
    pearson = pearsonr(pred, label)
    acc = accuracy_score(label, pred)
    prec = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    auc = roc_auc_score(label, prob, average='macro', multi_class='ovo')
    pred1 = torch.tensor(pred)
    label1 = torch.tensor(label)
    spec = Speci(pred1, label1)

    c1 = confusion_matrix(label, pred)
    print('icc:', icc)
    print('kappa1:', kappa1)
    print('kappa2', kappa2)
    print('pearson:', pearson)
    print('acc:', acc)
    print('prec:', prec)
    print('recall:', recall)
    print('f1:', f1)
    print('auc:', auc)
    print('spec:', spec)

    icc.loc[len(icc.index)] = ['kappa1', kappa1, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['kappa2', kappa2, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['pearson', pearson, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['acc', acc, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['prec', prec, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['recall', recall, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['f1', f1, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['auc', auc, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['spec', spec, '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['confusion_matrix', c1, '', '', '', '', '', '']

    predicted = np.array(pred)
    actual = np.array(label)
    print('confusion matrix:', c1)
    pred = np.where(predicted > 1, 1, 0)
    label = np.where(actual > 1, 1, 0)
    # 计算混淆矩阵
    print('ACC:', accuracy_score(label, pred))
    print('Precision:', precision_score(label, pred))
    print('Recall:', recall_score(label, pred))
    print('F1:', f1_score(label, pred))
    # print('specificity:', specificity(pred, label))
    print('AUC:', roc_auc_score(label, prob1))
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    print('specificity:', tn / (fp + tn))
    confusion_mat = np.array([[tp, fn], [fp, tn]])
    print('Confusion matrix: ', confusion_mat)

    icc.loc[len(icc.index)] = ['poor vs nonpoor', '', '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['ACC', accuracy_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Precision', precision_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Recall', recall_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['F1', f1_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['AUC', roc_auc_score(label, prob1), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Recall', recall_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['specificity', specificity(pred, label), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Confusion matrix', confusion_mat, '', '', '', '', '', '']

    pred = np.where(predicted > 0, 1, 0)
    label = np.where(actual > 0, 1, 0)
    # 计算混淆矩阵
    print('ACC:', accuracy_score(label, pred))
    print('Precision:', precision_score(label, pred))
    print('Recall:', recall_score(label, pred))
    print('F1:', f1_score(label, pred))
    # print('specificity:', specificity(pred, label))
    print('AUC:', roc_auc_score(label, prob2))
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    print('specificity:', tn / (fp + tn))
    confusion_mat = np.array([[tp, fn], [fp, tn]])
    print('Confusion matrix: ', confusion_mat)

    icc.loc[len(icc.index)] = ['good vs nongood', '', '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['ACC', accuracy_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Precision', precision_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Recall', recall_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['F1', f1_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['AUC', roc_auc_score(label, prob2), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Recall', recall_score(label, pred), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['specificity', specificity(pred, label), '', '', '', '', '', '']
    icc.loc[len(icc.index)] = ['Confusion matrix', confusion_mat, '', '', '', '', '', '']

    icc.to_csv(os.path.join(save_path, f'{cata}score.csv'), index=False)


def result(pred, label, prob1, prob2):
    predicted = np.array(pred)
    actual = np.array(label)
    c1 = confusion_matrix(label, pred)
    print('confusion matrix:', c1)
    print('3acc:', accuracy_score(label, pred))
    pred = np.where(predicted > 1, 1, 0)
    label = np.where(actual > 1, 1, 0)
    # 计算混淆矩阵
    print('poor vs nonpoor')
    print('poor_ACC:', accuracy_score(label, pred))
    print('Precision:', precision_score(label, pred))
    print('Recall:', recall_score(label, pred))
    print('F1:', f1_score(label, pred))
    # print('specificity:', specificity(pred, label))
    print('AUC:', roc_auc_score(label, prob1))
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    print('specificity:', tn / (fp + tn))
    confusion_mat = np.array([[tp, fn], [fp, tn]])
    print('Confusion matrix: ', confusion_matrix(label, pred))

    pred = np.where(predicted > 0, 1, 0)
    label = np.where(actual > 0, 1, 0)
    print('good vs nongood')
    print('ACC:', accuracy_score(label, pred))
    print('Precision:', precision_score(label, pred))
    print('Recall:', recall_score(label, pred))
    print('F1:', f1_score(label, pred))
    # print('specificity:', specificity(pred, label))
    print('AUC:', roc_auc_score(label, prob2))
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    print('specificity:', tn / (fp + tn))
    confusion_mat = np.array([[tp, fn], [fp, tn]])
    print('Confusion matrix: ', confusion_matrix(label, pred))


if __name__ == '__main__':
    fold = 3
    for i in range(3):
        model = mpvit_small().to(device)
        #model = DataParallel(model)
        print('Current fold: ', i + 1)
        train_set = ProVe(flag=0, fold=i, mode='train')
        val_set = ProVe(flag=0, fold=i, mode='test')
        tr_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
        v_loader = DataLoader(val_set, batch_size=1, num_workers=8, drop_last=False)
        #main(tr_loader, v_loader, i, model)
        torch.cuda.empty_cache()
        path1 = os.path.join(save_path, f'fold_{i}_model.pt')
        model.load_state_dict(torch.load(path1))
        pred, prob, label = evaluate1(v_loader, i, model)
        print(save_path)
        result(pred, label, prob[:, 2], 1 - prob[:, 0])
        torch.cuda.empty_cache()
        path2 = os.path.join(save_path, f'best_fold_{i}_model.pt')
        model.load_state_dict(torch.load(path2))
        pred, prob, label = evaluate2(v_loader, i, model)
        print(save_path)
        result(pred, label, prob[:, 2], 1 - prob[:, 0])
    score('', fold)
    score('best_', fold)
