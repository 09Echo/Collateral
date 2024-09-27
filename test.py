import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import warnings
from sklearn import metrics

from data_loader import ProVe
warnings.filterwarnings('ignore')
from method import mpvit_small
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_size", type=int, default=1, help="train batch size")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--checkpoint_path", type=str, default='./results', help = 'model_save_path')
    parser.add_argument("--checkpoint_name", default=['best_fold_0_model.pt','best_fold_1_model.pt','best_fold_2_model.pt','best_fold_3_model.pt','best_fold_4_model.pt'], help='model_name')
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()
    return args


def main(args):
    outprob = []
    p_y = []
    t_y = []
    for ki in range(args.k_fold):
        checkpoint_path = os.path.join(args.checkpoint_path, args.checkpoint_name[ki])
        model = mpvit_small(num_classes=args.num_classes).cuda()
        model.load_state_dict(torch.load(checkpoint_path))

        test_ds = ProVe(flag=0, fold=ki, mode='test', n_split=args.k_fold)
        test_dl = DataLoader(test_ds, batch_size=args.t_size, num_workers=4, drop_last=False)

        model.eval()
        with torch.no_grad():
            for data, label, _ in test_dl:
                data, label = data.to(args.device, dtype=torch.float), label.to(args.device)
                t_y.extend(label.tolist())
                outputs = model(data)
                predict_y = torch.max(outputs, dim=1)[1]
                p_y.extend(predict_y.tolist())

                prob = torch.softmax(outputs, dim=1)
                prob = prob.cpu().numpy()
                outprob.append(prob)
    outprob = np.vstack(outprob)

    precision, recall, test_f1, _ = metrics.precision_recall_fscore_support(t_y, p_y, average='macro')
    cm = metrics.confusion_matrix(t_y, p_y)
    ACC = metrics.accuracy_score(t_y, p_y)
    AUC = metrics.roc_auc_score(np.array(t_y), outprob, average='macro', multi_class='ovo')

    print(cm)
    print(ACC)
    print(AUC)


if __name__ == '__main__':
    args = parse_args()
    main(args)
