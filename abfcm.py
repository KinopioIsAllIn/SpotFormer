import os
import shutil
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from colorama import Fore
from colorama import Style
# from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F

from model import PEM, Model, MSSTGT
from datasets import LOSO_DATASET, CROSS_DATASET
from tool import save_model, save_model_per_subject, configure_optimizers
from post_process import calculate_proposal_with_score, nms, iou_for_find, iou_for_tp

from pathlib import Path

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='one',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, device=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=-1)
        class_mask = torch.nn.functional.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1)
        alpha = alpha.to(predict.device)

        probs = (pt * class_mask).sum(-1).view(-1, 1)
        log_p = probs.log()

        loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0,
                 reduction='mean'):
        super(MultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=-1).view(-1, self.class_num)
        class_mask = torch.nn.functional.one_hot(
            target, self.class_num).view(-1, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1)
        alpha = alpha.to(predict.device)

        positive_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 0).squeeze()
        negative_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 1).squeeze()
        positive_pt = pt[positive_class_mask_indices]
        negative_pt = pt[negative_class_mask_indices]
        positive_class_mask = class_mask[positive_class_mask_indices]
        negative_class_mask = class_mask[negative_class_mask_indices]
        positive_alpha = alpha[positive_class_mask_indices]
        negative_alpha = alpha[negative_class_mask_indices]

        # p_num = torch.sum(class_mask[:, :-1]).item()
        # n_num = torch.sum(class_mask[:, -1]).item()
        # if torch.sum(class_mask[:, -1]) == class_mask.shape[0]:
        #     return 0
        # negative_alpha = 1 / math.log2(n_num / p_num)
        # positive_alpha = 1 - negative_alpha

        positive_probs = (positive_pt * positive_class_mask).sum(-1).view(-1, 1)  # 只有属于自己标签的概率

        positive_log_p = positive_probs.log()
        positive_loss = -positive_alpha * torch.pow(
            (1 - positive_probs), self.gamma) * positive_log_p

        negative_probs = (negative_pt * negative_class_mask).sum(-1).view(-1, 1)
        negative_log_p = negative_probs.log()
        negative_loss = -negative_alpha * torch.pow(
            torch.clamp(1 - self.lb_smooth - negative_probs, min=0),
            self.gamma) * negative_log_p

        loss = torch.cat((positive_loss, negative_loss))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def _focal_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.view(-1)
    mask_class = (label > 0).float()

    c_1 = alpha
    c_0 = 1 - c_1
    loss = ((c_1 * torch.abs(label - output)**gamma * mask_class
            * torch.log(output + 0.00001))
            + (c_0 * torch.abs(label + lb_smooth - output)**gamma
            * (1.0 - mask_class)
            * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss


def _probability_loss(output, score, gamma, alpha, lb_smooth):
    output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss


def _l1_loss(output_distance, distance_start, distance_end, device):
    _loss = torch.nn.L1Loss()

    output_distance_start = output_distance[:, 0, :].contiguous().view(-1)
    output_distance_end = output_distance[:, 1, :].contiguous().view(-1)
    distance_start = distance_start.view(-1)
    distance_end = distance_end.view(-1)

    distance_start_indices = np.nonzero(distance_start).view(-1)
    distance_end_indices = np.nonzero(distance_end).view(-1)
    distance_start = distance_start[distance_start_indices]
    distance_end = distance_end[distance_end_indices]

    distance_start = distance_start.to(device)
    distance_end = distance_end.to(device)

    l_loss = _loss(output_distance_start[distance_start_indices],
                   distance_start)
    r_loss = _loss(output_distance_end[distance_end_indices], distance_end)
    return l_loss, r_loss


def _regression_loss(output_distance, distance_start, distance_end, device,
                     loss_type="smooth"):
    if loss_type == "mse":
        _loss = torch.nn.MSELoss()
    elif loss_type == "smooth":
        _loss = torch.nn.SmoothL1Loss()

    output_distance_start = output_distance[:, 0, :].contiguous().view(-1)
    output_distance_end = output_distance[:, 1, :].contiguous().view(-1)
    distance_start = distance_start.view(-1)
    distance_end = distance_end.view(-1)

    distance_start_indices = np.nonzero(distance_start).view(-1)
    distance_end_indices = np.nonzero(distance_end).view(-1)
    distance_start = distance_start[distance_start_indices]
    distance_end = distance_end[distance_end_indices]

    distance_start = distance_start.to(device)
    distance_end = distance_end.to(device)

    loss1 = _loss(output_distance_start[distance_start_indices],
                  distance_start)
    loss2 = _loss(output_distance_end[distance_end_indices], distance_end)
    return loss1 + loss2


def _spatial_pos_loss(spatial_pos_embedding):
    l1 = (F.mse_loss(spatial_pos_embedding[0, 0], spatial_pos_embedding[0, 1]) +
          F.mse_loss(spatial_pos_embedding[0, 1], spatial_pos_embedding[0, 2])) / 2
    l2 = (F.mse_loss(spatial_pos_embedding[0, 3], spatial_pos_embedding[0, 4]) +
          F.mse_loss(spatial_pos_embedding[0, 4], spatial_pos_embedding[0, 5])) / 2
    l3 = (F.mse_loss(spatial_pos_embedding[0, 6], spatial_pos_embedding[0, 7]) +
          F.mse_loss(spatial_pos_embedding[0, 7], spatial_pos_embedding[0, 8]) +
          F.mse_loss(spatial_pos_embedding[0, 8], spatial_pos_embedding[0, 9])) / 3
    return (l1+l2+l3)/3


def _train(model, data_loader, optimizer, epoch, device, writer, opt, subject=None):
    model.train(True)
    epoch_loss = 0

    for batch_idx, (feature, micro_apex_score, macro_apex_score,
                    micro_action_score, macro_action_score,
                    micro_start_end_label, macro_start_end_label, action_labels
                    ) in enumerate(data_loader):
        b, T, t, n, c = feature.shape
        feature = feature.reshape(-1, t, n, c)
        b, t, n, c = feature.shape

        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # including nose area
        # index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14]  # including eyelid area
        # index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]  # including all
        feature = feature[:, :, index, :]
        # index = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
        # feature = feature[:, :, index, :]


        feature = feature.permute(0, 3, 1, 2).contiguous()
        # feature = feature.reshape(b, t, -1).permute(0, 2, 1).contiguous() # for GCN
        feature = feature.to(device)

        micro_apex_score = micro_apex_score.to(device)
        macro_apex_score = macro_apex_score.to(device)
        micro_action_score = micro_action_score.to(device)
        macro_action_score = macro_action_score.to(device)
        micro_start_end_label = micro_start_end_label.to(device)
        macro_start_end_label = macro_start_end_label.to(device)

        STEP = int(opt["RECEPTIVE_FILED"] // 2)

        output_probability, output4contrastive, spatial_pos_embedding = model(feature)  # b, 10, M
        # output_probability = output_probability[:, :, STEP:-STEP]  # b, 10, 1

        # print(output_probability[:5])

        contrastive_loss = SupConLoss(temperature=0.5, base_temperature=0.07)
        action_labels = action_labels.to(device).reshape(-1)
        loss_contrastive = contrastive_loss(features=output4contrastive, labels=action_labels, device=device)

        # spatial_pos_loss = _spatial_pos_loss(spatial_pos_embedding)


        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_action = output_probability[:, 8, :]
        output_macro_action = output_probability[:, 9, :]

        output_micro_start_end = output_probability[:, 0: 0 + 3, :]
        output_macro_start_end = output_probability[:, 3: 3 + 3, :]

        loss_micro_apex = _probability_loss(output_micro_apex,
                                            micro_apex_score,
                                            opt["abfcm_apex_gamma"],
                                            opt["abfcm_apex_alpha"],
                                            opt["abfcm_label_smooth"])
        loss_macro_apex = _probability_loss(output_macro_apex,
                                            macro_apex_score,
                                            opt["abfcm_apex_gamma"],
                                            opt["abfcm_apex_alpha"],
                                            opt["abfcm_label_smooth"])
        loss_micro_action = _probability_loss(output_micro_action,
                                              micro_action_score,
                                              opt["abfcm_action_gamma"],
                                              opt["abfcm_action_alpha"],
                                              opt["abfcm_label_smooth"])
        loss_macro_action = _probability_loss(output_macro_action,
                                              macro_action_score,
                                              opt["abfcm_action_gamma"],
                                              opt["abfcm_action_alpha"],
                                              opt["abfcm_label_smooth"])

        _tmp_alpha = opt["abfcm_start_end_alpha"]
        _mul_focall_loss = MultiCEFocalLoss_New(
            class_num=3,
            alpha=torch.tensor(
                [_tmp_alpha / 2, _tmp_alpha / 2, 1 - _tmp_alpha],
                dtype=torch.float32),
            gamma=opt["abfcm_start_end_gama"],
            # lb_smooth=0.06,
        )

        # CELoss = torch.nn.CrossEntropyLoss()
        # output_micro_start_end = output_micro_start_end.squeeze()
        # micro_start_end_label = micro_start_end_label.reshape(-1)
        # loss_micro_start_end = CELoss(output_micro_start_end, micro_start_end_label)
        #
        # output_macro_start_end = output_macro_start_end.squeeze()
        # macro_start_end_label = macro_start_end_label.reshape(-1)
        # loss_macro_start_end = CELoss(output_macro_start_end, macro_start_end_label)

        CELoss = torch.nn.CrossEntropyLoss()
        output_micro_start_end = output_micro_start_end.permute(0, 2, 1).contiguous().reshape(-1, 3)
        micro_start_end_label = micro_start_end_label.reshape(-1)
        loss_micro_start_end = CELoss(output_micro_start_end, micro_start_end_label)

        output_macro_start_end = output_macro_start_end.permute(0, 2, 1).contiguous().reshape(-1, 3)
        macro_start_end_label = macro_start_end_label.reshape(-1)
        loss_macro_start_end = CELoss(output_macro_start_end, macro_start_end_label)

        # loss_micro_start_end = _mul_focall_loss(
        #     output_micro_start_end.permute(0, 2, 1).contiguous(),
        #     micro_start_end_label)
        # loss_macro_start_end = _mul_focall_loss(
        #     output_macro_start_end.permute(0, 2, 1).contiguous(),
        #     macro_start_end_label)

        loss = (1.8 * loss_micro_apex
                + 1.0 * loss_micro_start_end
                + 0.1 * loss_micro_action
                + opt['macro_ration'] * (
                    1.0 * loss_macro_apex
                    + 1.0 * loss_macro_start_end
                    + 0.1 * loss_macro_action
                )) + 0.005 * loss_contrastive  # + 0.005 * spatial_pos_loss

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(str(epoch) + ": " + str(epoch_loss))


def abfcm_train_and_eval(opt):
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    if opt['dataset'] == "cross":
        train_dataset = CROSS_DATASET(opt, "train")
        model = PEM(opt)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'],
                                               # drop_last=True,
                                               )

    optimizer = configure_optimizers(model, opt["abfcm_training_lr"],
                                     opt["abfcm_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, opt['abfcm_lr_scheduler'])

    def _result(type_idx):
        with torch.no_grad():
            _get_model_output_full(model, epoch, device, opt)
            abfcm_nms_once(opt, epoch)
            abfcm_iou_once(opt, epoch)

            anno_df = pd.read_csv(opt['anno_csv'])
            m = len(anno_df[anno_df['type_idx'] == type_idx])

            abfcm_final_result_csv = os.path.join(
                opt['project_root'], opt['output_dir_name'],
                'sub_abfcm_final_result', str(epoch).zfill(3) + '.csv')
            if os.path.exists(abfcm_final_result_csv) is not True:
                return 0, 0, 0, m

            find = 0
            tp = 0
            n = 0

            df = pd.read_csv(abfcm_final_result_csv)
            tmp_df = df[df["type_idx"] == type_idx]
            if len(tmp_df) > 0:
                find = len(tmp_df[tmp_df["find"] == True])
                tp = len(tmp_df[tmp_df["tp"] == True])
                n = len(tmp_df)

            return tp, find, n, m

    def _show_result(tp, find, n, m, type_idx=2):

        def _cal_result(tp, find, n, m):
            recall = 0
            precision = 0
            f1_score = 0
            recall = find / m
            if n > 0:
                precision = tp / n
            if n * find + m * tp > 0:
                f1_score = 2 * find / (n + m)
            return recall, precision, f1_score

        assert type_idx == 0 or type_idx == 1 or type_idx == 2, \
            "type_idx is invalid"
        if type_idx == 0:
            assert (type(tp) == tuple or type(tp) == list) and len(tp) == 2,\
                ("type of tp must be tuple or list and length of tp must be 2"
                 "when type_idx==2")
            assert (type(find) == tuple or type(find) == list) and len(find) == 2,\
                ("type of find must be tuple or list and length of find must be 2"
                 "when type_idx==2")
            assert (type(n) == tuple or type(n) == list) and len(n) == 2,\
                ("type of n must be tuple or list and length of n must be 2"
                 "when type_idx==2")
            assert (type(m) == tuple or type(m) == list) and len(m) == 2,\
                ("type of m must be tuple or list and length of m must be 2"
                 "when type_idx==2")
        type_str = "all" if type_idx == 0 else (
            "macro" if type_idx == 1 else "micro")

        if type_idx == 0:
            macro_recall, macro_precision, macro_f1_score = _cal_result(
                tp[0], find[0], n[0], m[0])
            micro_recall, micro_precision, micro_f1_score = _cal_result(
                tp[1], find[1], n[1], m[1])
            tp_all = tp[0] + tp[1]
            find_all = find[0] + find[1]
            n_all = n[0] + n[1]
            m_all = m[0] + m[1]

        recall_all, precision_all, f1_score_all = _cal_result(
            tp_all, find_all, n_all, m_all)

        print(f"[{str(epoch).zfill(3)}]-[{Fore.GREEN}macro{Style.RESET_ALL}]:",
              end=' ')
        print(f"tp: {str(tp[0]).zfill(3)}", end=' ')
        print(f"find: {str(find[0]).zfill(3)}", end=' ')
        print(f"predict_count: {str(n[0]).zfill(4)}", end=' ')
        print("precision: %.6f" % macro_precision, end=' ')
        print("recall: %.6f" % macro_recall, end=' ')
        print(f"f1_score: {Fore.GREEN}{macro_f1_score:.6f}{Style.RESET_ALL}")

        print(f"     -[{Fore.YELLOW}micro{Style.RESET_ALL}]:", end=' ')
        print(f"tp: {str(tp[1]).zfill(3)}", end=' ')
        print(f"find: {str(find[1]).zfill(3)}", end=' ')
        print(f"predict_count: {str(n[1]).zfill(4)}", end=' ')
        print("precision: %.6f" % micro_precision, end=' ')
        print("recall: %.6f" % micro_recall, end=' ')
        print(f"f1_score: {Fore.YELLOW}{micro_f1_score:.6f}{Style.RESET_ALL}")

        print(f"     -[{Fore.MAGENTA} all {Style.RESET_ALL}]:", end=' ')
        print(f"tp: {str(tp_all).zfill(3)}", end=' ')
        print(f"find: {str(find_all).zfill(3)}", end=' ')
        print(f"predict_count: {str(n_all).zfill(4)}", end=' ')
        print("precision: %.6f" % precision_all, end=' ')
        print("recall: %.6f" % recall_all, end=' ')
        print(f"f1_score: {Fore.MAGENTA}{f1_score_all:.6f}{Style.RESET_ALL}")
        print()
        return ((macro_recall, micro_recall, recall_all),
                (macro_precision, micro_precision, precision_all),
                (macro_f1_score, micro_f1_score, f1_score_all)
                )

    epoch = -1
    print(f"length of train_dataset: {len(train_dataset)}")
    print("result before training:")
    macro_tp, macro_find, macro_n, macro_m = _result(type_idx=1)
    micro_tp, micro_find, micro_n, micro_m = _result(type_idx=2)
    _show_result((macro_tp, micro_tp),
                 (macro_find, micro_find),
                 (macro_n, micro_n),
                 (macro_m, micro_m),
                 type_idx=0)
    print()

    writer = SummaryWriter()

    for epoch in range(opt['epochs']):
        _train(model, train_loader, optimizer, epoch, device, None, opt)
        scheduler.step()
        if opt['verbose']:
            print("lr: ", scheduler.get_last_lr())
        if opt['save_model'] is True:
            save_model(opt["model_save_root"], "abfcm_models",
                       epoch, model, optimizer)
        macro_tp, macro_find, macro_n, macro_m = _result(type_idx=1)
        micro_tp, micro_find, micro_n, micro_m = _result(type_idx=2)

        ((macro_recall, micro_recall, recall_all),
         (macro_precision, micro_precision, precision_all),
         (macro_f1_score, micro_f1_score, f1_score_all)
         ) = _show_result(
            (macro_tp, micro_tp),
            (macro_find, micro_find),
            (macro_n, micro_n),
            (macro_m, micro_m),
            type_idx=0)
        writer.add_scalar("macro_f1_score", macro_f1_score, epoch)
        writer.add_scalar("micro_f1_score", micro_f1_score, epoch)
        writer.add_scalar("f1_score_all", f1_score_all, epoch)
    writer.close()


def abfcm_train(opt, subject=None):
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    train_dataset = LOSO_DATASET(opt, "train", subject)

    print("Data: %d." % len(train_dataset))

    # model = PEM(opt)
    # model = Model(in_channels=2, num_class=10, edge_importance_weighting=True)
    hidden_dim = 256  # 256 best
    model = MSSTGT(
        seq_len=opt["RECEPTIVE_FILED"],
        num_classes=10,
        dim=hidden_dim,
        depth=4,
        heads=4,
        mlp_dim=hidden_dim,
        dropout=0.1,
        emb_dropout=0.1)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])

    optimizer = configure_optimizers(model, opt["abfcm_training_lr"],
                                     opt["abfcm_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, opt['abfcm_lr_scheduler'])

    for epoch in range(opt['epochs']):
        _train(model, train_loader, optimizer, epoch, device, None, opt,
               subject)
        scheduler.step()
        if opt['verbose']:
            print("lr: ", scheduler.get_last_lr())
        if opt['save_model'] is True:
            save_model_per_subject(opt["model_save_root"], "abfcm_models",
                                   epoch, model, optimizer, subject)


def abfcm_train_group(opt, ca_subject):
    for subject in ca_subject:
        start_time = datetime.now()

        abfcm_train(opt, subject)

        cur_time = datetime.now()
        delta_time = cur_time - start_time
        print(f"【{subject}-Time】: {delta_time}")


def abfcm_output(opt, subject):
    device = opt['device'] if torch.cuda.is_available() else 'cpu'

    # model = PEM(opt)
    # model = Model(2, 10, True)
    hidden_dim = 256  # 256 best
    model = MSSTGT(
        seq_len=opt["RECEPTIVE_FILED"],
        num_classes=10,
        dim=hidden_dim,
        depth=4,
        heads=4,
        mlp_dim=hidden_dim,
        dropout=0.1,
        emb_dropout=0.1)

    model = model.to(device)
    epoch_begin = opt['epoch_begin']
    for epoch in range(opt['epochs']):
        if epoch >= epoch_begin:
            with torch.no_grad():
                if opt['verbose']:
                    print(f"【{epoch}-{subject}】:", end=' ')
                weight_file = os.path.join(
                    opt["model_save_root"], subject, "abfcm_models",
                    "model_" + str(epoch).zfill(3) + ".pth")
                checkpoint = torch.load(weight_file,
                                        map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint['model'])
                _get_model_output_full(model, epoch, device, opt,
                                       split="test", subject=subject)


def abfcm_output_group(opt, ca_subject):
    for subject in ca_subject:
        start_time = datetime.now()

        abfcm_output(opt, subject)

        cur_time = datetime.now()
        delta_time = cur_time - start_time
        print(f"【{subject}-Time】: {delta_time}")


def abfcm_nms_once(opt, epoch, subject=None):
    if subject is not None:
        abfcm_nms_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            subject, 'abfcm_nms',
            str(epoch).zfill(3) + '.csv')
        abfcm_out_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            subject, "abfcm_out",
            str(epoch).zfill(3) + ".csv")
    else:
        abfcm_nms_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            'abfcm_nms',
            str(epoch).zfill(3) + '.csv')
        dir_path, _ = os.path.split(abfcm_nms_csv)
        if os.path.exists(dir_path) is not True:
            os.makedirs(dir_path)
        abfcm_out_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            "abfcm_out",
            str(epoch).zfill(3) + ".csv")

    if os.path.exists(abfcm_nms_csv):
        os.remove(abfcm_nms_csv)

    if os.path.exists(abfcm_out_csv):
        df = pd.read_csv(abfcm_out_csv)
        # df["tp_fp"] = np.array(["FP"]*len(df))
        df = df.groupby(['video_name', "type_idx"], group_keys=True).apply(
            lambda x: nms(x, opt)).reset_index(drop=True)
        if len(df) > 0:
            df.to_csv(abfcm_nms_csv, index=False)


def abfcm_nms(opt, subject):
    epoch_begin = opt['epoch_begin']

    for epoch in range(epoch_begin, opt['epochs']):
        abfcm_nms_once(opt, epoch, subject)


def abfcm_iou_once(opt, epoch, subject=None):
    if subject is not None:
        abfcm_final_result_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'], subject,
            'sub_abfcm_final_result', str(epoch).zfill(3) + '.csv')
        nms_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            subject, "abfcm_nms", str(epoch).zfill(3) + ".csv")
    else:
        abfcm_final_result_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            'sub_abfcm_final_result', str(epoch).zfill(3) + '.csv')
        dir_path, _ = os.path.split(abfcm_final_result_csv)
        if os.path.exists(dir_path) is not True:
            os.makedirs(dir_path)
        nms_csv = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            "abfcm_nms", str(epoch).zfill(3) + ".csv")

    if os.path.exists(abfcm_final_result_csv):
        os.remove(abfcm_final_result_csv)

    new_df = pd.DataFrame()
    if os.path.exists(nms_csv):
        nms_df = pd.read_csv(nms_csv)
        new_df = nms_df.groupby(['video_name', "type_idx"], group_keys=True).apply(
            lambda x: iou_for_find(x, opt)).reset_index(drop=True)
        new_df = new_df.groupby(['video_name', "type_idx"], group_keys=True).apply(
            lambda x: iou_for_tp(x, opt)).reset_index(drop=True)

    if len(new_df) > 0:
        new_df.to_csv(abfcm_final_result_csv, index=False)
    return new_df


def abfcm_iou_process(opt, subject=None):
    def _cal_metrics_for_show(tp_find_df, type_idx):
        if len(tp_find_df) == 0:
            return (0, 0, 0, 0, 0)

        find = 0
        tp = 0
        n = 0
        m = 0
        recall = 0
        precision = 0
        f1_score = 0
        tmp_df = tp_find_df[tp_find_df["type_idx"] == type_idx]
        if len(tmp_df) > 0:
            find = len(tmp_df[tmp_df["find"] == "True"])
            tp = len(tmp_df[tmp_df["tp"] == "True"])
            n = len(tmp_df)
            tmp_anno_df = anno_df[anno_df['subject'] == subject]
            tmp_anno_df = tmp_anno_df[tmp_anno_df['type_idx'] == type_idx]
            m = len(tmp_anno_df)

        find += 0.01
        tp += 0.01
        if m > 0:
            recall = find / m
        if n > 0:
            precision = find / n
        if n * find + m * find > 0:
            f1_score = (2 * find) / (n + m)
        return (find, tp, recall, precision, f1_score)

    epoch_begin = opt['epoch_begin']
    anno_df = pd.read_csv(opt['anno_csv'])

    for epoch in range(epoch_begin, opt['epochs']):
        new_df = abfcm_iou_once(opt, epoch, subject)

def _get_model_output_full(model, epoch, device, opt,
                           split="test", subject=None):
    def _cal_proposal(array_softmax_score, array_apex_score, video_len,
                      video_name, type_idx):
        array_score_micro_start = array_softmax_score[:, 0, :].squeeze()
        array_score_micro_end = array_softmax_score[:, 1, :].squeeze()
        # array_score_micro_none = softmax_score[:,2,:].squeeze()

        proposal_block = calculate_proposal_with_score(
            array_score_micro_start, array_score_micro_end,
            array_apex_score, video_len, type_idx, opt)
        if proposal_block is not None:
            array_video_name = np.array(
                [video_name] * len(proposal_block)).reshape(-1, 1)
            array_type_idx = np.array(
                [type_idx] * len(proposal_block)).reshape(-1, 1)
            proposal_block = np.concatenate(
                (array_video_name, proposal_block, array_type_idx), axis=1)
        return proposal_block

    model.eval()

    if subject is not None:
        predict_file = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            subject, 'abfcm_out', str(epoch).zfill(3) + '.csv')
        feature_path_list = glob.glob(os.path.join(
            opt["feature_root"], subject, "*.npy"))
    else:
        predict_file = os.path.join(
            opt['project_root'], opt['output_dir_name'],
            'abfcm_out', str(epoch).zfill(3) + '.csv')
        dir_path, _ = os.path.split(predict_file)
        if os.path.exists(dir_path) is not True:
            os.makedirs(dir_path)
        feature_path_list = []
        for item in Path(os.path.join(opt["feature_root"], split)).iterdir():
            feature_path_list += glob.glob(os.path.join(
                str(item), "*.npy"))

    if os.path.exists(predict_file):
        os.remove(predict_file)

    tmp_array = []

    col_name = ["video_name", "start_frame", "end_frame", "apex_frame", "start_score",
                "end_score", "apex_score", "type_idx"]
    for feature_path in feature_path_list:
        feature = np.load(feature_path)
        video_name = feature_path.split('/')[-1].split('\\')[-1].split(".")[0][:13]  # [:13] for casme
        feature = torch.from_numpy(feature).float()
        T, t, n, c = feature.shape

        index = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12]

        feature = feature[:, :, index, :]

        # for GCN
        feature = feature.permute(0, 3, 1, 2).contiguous()
        feature = feature.to(device)

        # video_len = t
        video_len = T
        STEP = int(opt["RECEPTIVE_FILED"] // 2)
        output_probability, feature_map, spatial_pos = model(feature)

        output_probability = output_probability.permute(2, 1, 0)  # 1, output, T

        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_start_end = output_probability[:, 0: 0 + 3, :]
        output_macro_start_end = output_probability[:, 3: 3 + 3, :]

        # micro expression
        array_softmax_score_micro = torch.softmax(
            output_micro_start_end.cpu(), dim=1).numpy()
        array_score_micro_apex = torch.sigmoid(
            output_micro_apex.cpu()).squeeze().numpy()
        micro_proposal_block = _cal_proposal(
            array_softmax_score_micro, array_score_micro_apex, video_len,
            video_name, type_idx=2)
        if micro_proposal_block is not None:
            tmp_array.append(micro_proposal_block)

        # macro expression
        array_softmax_score_macro = torch.softmax(
            output_macro_start_end.cpu(), dim=1).numpy()
        array_score_macro_apex = torch.sigmoid(
            output_macro_apex.cpu()).squeeze().numpy()
        macro_proposal_block = _cal_proposal(
            array_softmax_score_macro, array_score_macro_apex, video_len,
            video_name, type_idx=1)
        if macro_proposal_block is not None:
            tmp_array.append(macro_proposal_block)

    if len(tmp_array) > 0:
        tmp_array = np.concatenate(tmp_array, axis=0)
        new_df = pd.DataFrame(tmp_array, columns=col_name)
        # TODO: the score should be a weighted sum
        new_df["score"] = new_df.start_score * new_df.end_score * new_df.apex_score
        # new_df["score"] = new_df.action_score
        new_df = new_df.groupby('video_name', group_keys=True).apply(
            lambda x: x.sort_values("score", ascending=False)).reset_index(drop=True)

        if len(new_df) > 0:
            new_df.to_csv(predict_file, index=False)


def abfcm_final_result_per_subject(opt, subject_list, type_idx=2):
    if type_idx == 0:
        print("spotting macro and micro expression")
    elif type_idx == 1:
        print("only spotting macro expression")
    elif type_idx == 2:
        print("only spotting micro expression")
    else:
        raise f"type_idx: {type_idx} is invalid value"

    def _cal_metrics(csv_path, type_idx):
        find = 0
        tp = 0
        m = 0
        n = 0
        f1_score = 0

        all_IoU = 0.0
        all_w = 0

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            micro_df = df[df["type_idx"] == type_idx]
            if len(micro_df):
                find += len(micro_df[micro_df["find"] == True])
                tp += len(micro_df[micro_df["tp"] == True])
                n += len(micro_df)

                # print(micro_df[micro_df["find"] == True]["iou"])
                all_IoU += micro_df[micro_df["find"] == True]["wiou"].astype(np.float32).sum().item()
                all_w += micro_df[micro_df["find"] == True]["w"].astype(np.float32).astype(np.int_).sum().item()

        tmp_df = anno_df[anno_df['subject'] == subject]
        tmp_df = tmp_df[tmp_df['type_idx'] == type_idx]
        m += len(tmp_df)

        find += 0.01
        tp += 0.01
        if n * find + m * tp > 0:
            f1_score = 2 * find / (n + m)
        return find, tp, n, m, f1_score, all_IoU, all_w

    log_file_path = './log_samm25'
    if os.path.exists(log_file_path):
        shutil.rmtree(log_file_path)
    # writer = SummaryWriter(log_file_path)
    col_name = ["subject", "best_epoch", "best_f1_score", "tp", "find", "n", "wiou", "w"]

    epoch_begin = opt['epoch_begin']

    anno_df = pd.read_csv(opt['anno_csv'])
    subject_best_epoch_list = []
    subject_best_f1_score_list = []
    subject_best_tp_list = []
    subject_best_find_list = []
    subject_best_n_list = []
    subject_best_iou_list = []
    subject_best_w_list = []


    for subject in subject_list:
        f1_score_list = []
        tp_list = []
        find_list = []
        n_list = []
        m_list = []
        iou_list = []
        w_list = []
        index_record_list = []
        for epoch in range(epoch_begin, opt['epochs']):
            csv_path = os.path.join(
                opt['project_root'], opt['output_dir_name'],
                subject, "sub_abfcm_final_result",
                str(epoch).zfill(3) + ".csv")

            if type_idx == 0:
                find, tp, n, m, f1_score, wiou, w = 0, 0, 0, 0, 0, 0.0, 0
                tmp_find, tmp_tp, tmp_n, tmp_m, tmp_f1_score, tmp_iou, tmp_w = _cal_metrics(
                    csv_path, type_idx=1)
                find += tmp_find
                tp += tmp_tp
                n += tmp_n
                m += tmp_m
                wiou += tmp_iou
                w += tmp_w
                # f1_score += tmp_f1_score
                tmp_find, tmp_tp, tmp_n, tmp_m, tmp_f1_score, tmp_iou, tmp_w = _cal_metrics(
                    csv_path, type_idx=2)
                find += tmp_find
                tp += tmp_tp
                n += tmp_n
                m += tmp_m
                # f1_score += tmp_f1_score
                # f1_score = 2 * find / (n+m)
                if (n+m) == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (find) / (n + m)
                wiou += tmp_iou
                w += tmp_w

            else:
                find, tp, n, m, f1_score, wiou, w = _cal_metrics(csv_path, type_idx)

            f1_score_list.append(f1_score)
            tp_list.append(tp)
            find_list.append(find)
            n_list.append(n)
            m_list.append(m)
            iou_list.append(wiou)
            w_list.append(w)
            # writer.add_scalars('f1', {subject: f1_score}, epoch)

        best_index = f1_score_list.index(max(f1_score_list))
        _max_f1_score = f1_score_list[best_index]
        _max_tp = tp_list[best_index]
        _max_find = find_list[best_index]
        best_n = n_list[best_index]
        _max_iou = iou_list[best_index]
        _max_w = w_list[best_index]

        subject_best_epoch_list.append(best_index + epoch_begin)
        subject_best_f1_score_list.append(_max_f1_score)
        subject_best_tp_list.append(_max_tp)
        subject_best_find_list.append(_max_find)
        subject_best_n_list.append(best_n)
        subject_best_iou_list.append(_max_iou)
        subject_best_w_list.append(_max_w)

    tmp_array = np.array((subject_list, subject_best_epoch_list,
                          subject_best_f1_score_list, subject_best_tp_list,
                          subject_best_find_list, subject_best_n_list, subject_best_iou_list, subject_best_w_list)).T
    new_df = pd.DataFrame(tmp_array, columns=col_name)
    best_epoch_csv = os.path.join(opt['project_root'], opt['output_dir_name'],
                                  f'type_idx_{type_idx}_best_epoch_csv.csv')
    if os.path.exists(best_epoch_csv):
        os.remove(best_epoch_csv)
    new_df.to_csv(best_epoch_csv, index=False)
    TP = new_df.tp.values.astype(np.float32).astype(np.int32).sum().item()
    FIND = new_df.find.values.astype(np.float32).astype(np.int32).sum().item()
    N = new_df.n.values.astype(np.float32).astype(np.int32).sum().item()
    IOU = new_df.wiou.values.astype(np.float32).sum().item()
    W = new_df.w.values.astype(np.float32).astype(np.int32).sum().item()

    print(IOU, FIND, W)

    if type_idx == 0:
        M = len(anno_df)
    else:
        M = len(anno_df[anno_df['type_idx'] == type_idx])

    print("【TP】 = %d" % TP)
    print("【Find】 = %d" % FIND)
    print("【Predict_Count】 = %d" % N)
    print("【TRUE ME COUNT】 = %d" % M)
    print("【F1_Score】= %.6f" % (2 * (FIND) / (N + M)))
    print("【IOU】= %.6f" % (IOU/W))
    print("\n")
    # writer.close()


def abfcm_final_result_best(opt, subject_list, type_idx=2):
    anno_df = pd.read_csv(opt['anno_csv'])

    best_epoch_csv = os.path.join(opt['project_root'], opt['output_dir_name'],
                                  f'type_idx_{type_idx}_best_epoch_csv.csv')
    best_epoch_df = pd.read_csv(best_epoch_csv)

    def _cal_metrics(type_idx):
        find = 0
        tp = 0
        m = 0
        n = 0
        apex_diff = 0.0
        for subject in subject_list:
            # TODO: directory name of cas(ME) and samm is not compatible
            epoch = best_epoch_df[
                best_epoch_df["subject"] == subject]["best_epoch"].item()
            csv_path = os.path.join(
                opt['project_root'], opt['output_dir_name'],
                subject, "sub_abfcm_final_result",
                str(epoch).zfill(3) + ".csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = df[df["type_idx"] == type_idx]
                find += len(df[df["find"] == True])
                tp += len(df[df["tp"] == True])
                n += len(df)

                apex_diff += sum(df["apex_diff"].values)

            tmp_df = anno_df[anno_df['subject'] == subject]
            tmp_df = tmp_df[tmp_df['type_idx'] == type_idx]
            m += len(tmp_df)
        precision = 0
        recall = 0
        f1_score = 0

        if m > 0:
            recall = find / m
        if n > 0:
            precision = find / n
        if n * find + m * tp > 0:
            f1_score = 2 * find / (n + m)
        return find, tp, n, recall, f1_score, precision, apex_diff

    (macro_find, macro_tp, macro_n, macro_recall, macro_f1_score,
        macro_precision, macro_apex_diff) = _cal_metrics(type_idx=1)
    (micro_find, micro_tp, micro_n, micro_recall, micro_f1_score,
        micro_precision, micro_apex_diff) = _cal_metrics(type_idx=2)

    print("================ FINAL RESULT ================")
    print("MACRO RESULT")
    print("【MACRO_F1_Score】= %.6f" % (macro_f1_score))
    print("【MACRO_Precision】= %.6f" % (macro_precision))
    print("【MACRO_Recall】= %.6f" % (macro_recall))
    print("【MACRO_TP】 = %d" % macro_tp)
    print("【MACRO_Find】 = %d" % macro_find)
    print("【MACRO_Predict_Count】 = %d" % macro_n)
    print("【MACRO_Apex_Diff】 = %f" % (macro_apex_diff/macro_find))

    print("\nMICRO RESULT")
    print("【MICRO_F1_Score】= %.6f" % (micro_f1_score))
    print("【MICRO_Precision】= %.6f" % (micro_precision))
    print("【MICRO_Recall】= %.6f" % (micro_recall))
    print("【MICRO_TP】 = %d" % micro_tp)
    print("【MICRO_Find】 = %d" % micro_find)
    print("【MICRO_Predict_Count】 = %d" % micro_n)
    print("【MICRO_Apex_Diff】 = %f" % (micro_apex_diff/micro_find))

    print("【Overall_Apex_Diff】 = %f" % ((macro_apex_diff + micro_apex_diff)/(macro_find+micro_find)))



def generate_proposal(opt, model, epoch):
    pass
