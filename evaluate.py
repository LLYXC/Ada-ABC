import torch
import numpy as np
from util import MultiDimAverageMeter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, roc_curve
import torch.nn as nn
from tqdm import tqdm


def evaluate_acc_ap_auc(model, data_loader, attr_dims, args, acc_only=False, drain=False, is_validation=False, threshold=0.5):
    device = args.device
    target_attr_idx = args.target_attr_idx
    bias_attr_idx = args.bias_attr_idx
    model.eval()
    gts = torch.LongTensor().to(device)
    bias_gts = torch.LongTensor().to(device)
    probs = torch.FloatTensor().to(device)
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    for index, path, data, attr, mask in tqdm(data_loader, leave=False):
        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]
        data = data.to(device)
        attr = attr.to(device)
        label = label.to(device)
        bias_label = bias_label.to(device)
        with torch.no_grad():
            logit = model(data)
            prob = torch.softmax(logit, dim=1)
            pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()
        gts = torch.cat((gts, label), 0)
        bias_gts = torch.cat((bias_gts, bias_label), 0)
        probs = torch.cat((probs, prob), 0)
        attr = attr[:, [target_attr_idx, bias_attr_idx]]
        attrwise_acc_meter.add(correct.cpu(), attr.cpu())
    accs = attrwise_acc_meter.get_mean()
    gts_numpy = gts.cpu().detach().numpy()
    probs_numpy = probs.cpu().detach().numpy()
    bias_gts_numpy = bias_gts.cpu().detach().numpy()
    if not acc_only:
        aps, aucs = [], []
        # overall auc and ap
        if drain:
            # For validation set, compute optimal threshold
            if is_validation:
                fpr, tpr, thresholds = roc_curve(gts_numpy, probs_numpy[:, 1])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                threshold = optimal_threshold
                pred_numpy = (probs_numpy[:, 1] >= optimal_threshold).astype(int)
            else:
                # For test set, use provided threshold
                pred_numpy = (probs_numpy[:, 1] >= threshold).astype(int)

            # Overall metrics
            aps.append(average_precision_score(gts_numpy, probs_numpy[:, 1]))
            aucs.append(roc_auc_score(gts_numpy, probs_numpy[:, 1])) # overall
            accuracy = accuracy_score(gts_numpy, pred_numpy)
            f1 = f1_score(gts_numpy, pred_numpy)
            precision = precision_score(gts_numpy, pred_numpy)
            recall = recall_score(gts_numpy, pred_numpy)

            metrics = {}

            metrics["overall/auc"] = aucs[-1]
            metrics["overall/ap"] = aps[-1]
            metrics["overall/acc"] = accuracy
            metrics["overall/f1"] = f1
            metrics["overall/precision"] = precision
            metrics["overall/recall"] = recall

            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 1)) # pneumo-without-drain
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 1)) # pneumo-with-drain
            idx3 = np.where((bias_gts_numpy == 0) & (gts_numpy == 0)) # neg

            # chest-drained metrics
            chest_drained_gts = np.concatenate([gts_numpy[idx2], gts_numpy[idx3]])
            chest_drained_probs = np.concatenate([probs_numpy[idx2][:, 1], probs_numpy[idx3][:, 1]])
            chest_drained_preds = (chest_drained_probs >= threshold).astype(int)
            aucs.append(roc_auc_score(chest_drained_gts, chest_drained_probs))
            chest_drained_accuracy = accuracy_score(chest_drained_gts, chest_drained_preds)
            chest_drained_f1 = f1_score(chest_drained_gts, chest_drained_preds)
            chest_drained_precision = precision_score(chest_drained_gts, chest_drained_preds)
            chest_drained_recall = recall_score(chest_drained_gts, chest_drained_preds)

            metrics["chest_drained/auc"] = aucs[-1]
            metrics["chest_drained/ap"] = aps[-1]
            metrics["chest_drained/acc"] = chest_drained_accuracy
            metrics["chest_drained/f1"] = chest_drained_f1
            metrics["chest_drained/precision"] = chest_drained_precision
            metrics["chest_drained/recall"] = chest_drained_recall

            # chest-w/o-drain metrics
            chest_wo_drain_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx3]])
            chest_wo_drain_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx3][:, 1]])
            chest_wo_drain_preds = (chest_wo_drain_probs >= threshold).astype(int)
            aucs.append(roc_auc_score(chest_wo_drain_gts, chest_wo_drain_probs))
            chest_wo_drain_accuracy = accuracy_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_f1 = f1_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_precision = precision_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_recall = recall_score(chest_wo_drain_gts, chest_wo_drain_preds)

            metrics["chest_wo_drained/auc"] = aucs[-1]
            metrics["chest_wo_drained/ap"] = aps[-1]
            metrics["chest_wo_drained/acc"] = chest_wo_drain_accuracy
            metrics["chest_wo_drained/f1"] = chest_wo_drain_f1
            metrics["chest_wo_drained/precision"] = chest_wo_drain_precision
            metrics["chest_wo_drained/recall"] = chest_wo_drain_recall



        else:
            # For validation set, compute optimal threshold
            if is_validation:
                fpr, tpr, thresholds = roc_curve(gts_numpy, probs_numpy[:, 1])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                threshold = optimal_threshold
                pred_numpy = (probs_numpy[:, 1] >= optimal_threshold).astype(int)
            else:
                # For test set, use provided threshold
                pred_numpy = (probs_numpy[:, 1] >= threshold).astype(int)

            # Create metrics dictionary
            metrics = {}

            # Overall metrics
            aps.append(average_precision_score(gts_numpy, probs_numpy[:, 1]))
            aucs.append(roc_auc_score(gts_numpy, probs_numpy[:, 1]))
            metrics["overall/auc"] = aucs[-1]
            metrics["overall/ap"] = aps[-1]
            metrics["overall/acc"] = accuracy_score(gts_numpy, pred_numpy)
            metrics["overall/f1"] = f1_score(gts_numpy, pred_numpy)
            metrics["overall/precision"] = precision_score(gts_numpy, pred_numpy)
            metrics["overall/recall"] = recall_score(gts_numpy, pred_numpy)

            # Aligned metrics
            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 0))
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 1))
            aligned_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx2]])
            aligned_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])
            aligned_preds = (aligned_probs >= threshold).astype(int)
            
            aps.append(average_precision_score(aligned_gts, aligned_probs))
            aucs.append(roc_auc_score(aligned_gts, aligned_probs))
            metrics["aligned/auc"] = aucs[-1]
            metrics["aligned/ap"] = aps[-1]
            metrics["aligned/acc"] = accuracy_score(aligned_gts, aligned_preds)
            metrics["aligned/f1"] = f1_score(aligned_gts, aligned_preds)
            metrics["aligned/precision"] = precision_score(aligned_gts, aligned_preds)
            metrics["aligned/recall"] = recall_score(aligned_gts, aligned_preds)

            # Conflict metrics
            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 1))
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 0))
            conflict_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx2]])
            conflict_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])
            conflict_preds = (conflict_probs >= threshold).astype(int)

            aps.append(average_precision_score(conflict_gts, conflict_probs))
            aucs.append(roc_auc_score(conflict_gts, conflict_probs))
            metrics["conflict/auc"] = aucs[-1]
            metrics["conflict/ap"] = aps[-1]
            metrics["conflict/acc"] = accuracy_score(conflict_gts, conflict_preds)
            metrics["conflict/f1"] = f1_score(conflict_gts, conflict_preds)
            metrics["conflict/precision"] = precision_score(conflict_gts, conflict_preds)
            metrics["conflict/recall"] = recall_score(conflict_gts, conflict_preds)

            # balanced metrics
            # Balanced metrics (average of aligned and conflict)
            metrics["balanced/auc"] = (metrics["aligned/auc"] + metrics["conflict/auc"]) / 2
            metrics["balanced/ap"] = (metrics["aligned/ap"] + metrics["conflict/ap"]) / 2
            metrics["balanced/acc"] = (metrics["aligned/acc"] + metrics["conflict/acc"]) / 2
            metrics["balanced/f1"] = (metrics["aligned/f1"] + metrics["conflict/f1"]) / 2
            metrics["balanced/precision"] = (metrics["aligned/precision"] + metrics["conflict/precision"]) / 2
            metrics["balanced/recall"] = (metrics["aligned/recall"] + metrics["conflict/recall"]) / 2
            aps.append((metrics["aligned/ap"] + metrics["conflict/ap"]) / 2)
            aucs.append((metrics["aligned/auc"] + metrics["conflict/auc"]) / 2)

        model.train()
        if is_validation:
            return metrics, optimal_threshold
        else:
            return metrics
    else:
        return accs

    
def evaluate_acc_ap_auc_mhead(model, data_loader, attr_dims, args, acc_only=False, drain=False, is_validation=False, threshold=0.5):

    model_tag = args.model_tag
    device = args.device
    target_attr_idx = args.target_attr_idx
    bias_attr_idx = args.bias_attr_idx
    num_heads = args.num_heads


    model.eval()
    gts = torch.LongTensor().to(device)
    bias_gts = torch.LongTensor().to(device)
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    probs = torch.FloatTensor().to(device)

    for index, path, data, attr, mask in tqdm(data_loader, leave=False):
        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]
        data = data.to(device)
        attr = attr.to(device)
        label = label.to(device)
        bias_label = bias_label.to(device)
        with torch.no_grad():
            if model_tag == 'ResNet18':
                feature = model.features(data)
                feature = feature.view(feature.size(0), -1).to(device)
            elif model_tag == 'DenseNet121' and args.xrv_weight:
                feature = model.features2(data)
            elif model_tag == 'DenseNet121':
                feature = model.features(data).to(device)
                feature = nn.AdaptiveAvgPool2d((1, 1))(feature)
                feature = torch.flatten(feature, 1)
            logit = []
            prob = []
            for i in range(num_heads):
                if model_tag == 'DenseNet121':
                    l = model.classifiers[i](feature)
                elif model_tag == 'ResNet18':
                    l = model.classifiers[i](feature)
                p = torch.softmax(l, dim=1)
                logit.append(l)
                prob.append(p)       
            prob = torch.mean(torch.stack(prob), dim=0)
            pred = prob.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()

        gts = torch.cat((gts, label), 0)
        bias_gts = torch.cat((bias_gts, bias_label), 0)
        probs = torch.cat((probs, prob), 0)
        attr = attr[:, [target_attr_idx, bias_attr_idx]]
        attrwise_acc_meter.add(correct.cpu(), attr.cpu())

    attrwise_acc_meter.add(correct.cpu(), attr.cpu())
    accs = attrwise_acc_meter.get_mean()
    gts_numpy = gts.cpu().detach().numpy()
    probs_numpy = probs.cpu().detach().numpy()
    bias_gts_numpy = bias_gts.cpu().detach().numpy()

    if not acc_only:
        aps, aucs = [], []
        # overall auc and ap
        if drain:
            # For validation set, compute optimal threshold
            if is_validation:
                fpr, tpr, thresholds = roc_curve(gts_numpy, probs_numpy[:, 1])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                threshold = optimal_threshold
                pred_numpy = (probs_numpy[:, 1] >= optimal_threshold).astype(int)
            else:
                # For test set, use provided threshold
                pred_numpy = (probs_numpy[:, 1] >= threshold).astype(int)

            # Overall metrics
            aps.append(average_precision_score(gts_numpy, probs_numpy[:, 1]))
            aucs.append(roc_auc_score(gts_numpy, probs_numpy[:, 1])) # overall
            accuracy = accuracy_score(gts_numpy, pred_numpy)
            f1 = f1_score(gts_numpy, pred_numpy)
            precision = precision_score(gts_numpy, pred_numpy)
            recall = recall_score(gts_numpy, pred_numpy)

            metrics = {}

            metrics["overall/auc"] = aucs[-1]
            metrics["overall/ap"] = aps[-1]
            metrics["overall/acc"] = accuracy
            metrics["overall/f1"] = f1
            metrics["overall/precision"] = precision
            metrics["overall/recall"] = recall

            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 1)) # pneumo-without-drain
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 1)) # pneumo-with-drain
            idx3 = np.where((bias_gts_numpy == 0) & (gts_numpy == 0)) # neg

            # chest-drained metrics
            chest_drained_gts = np.concatenate([gts_numpy[idx2], gts_numpy[idx3]])
            chest_drained_probs = np.concatenate([probs_numpy[idx2][:, 1], probs_numpy[idx3][:, 1]])
            chest_drained_preds = (chest_drained_probs >= threshold).astype(int)
            aucs.append(roc_auc_score(chest_drained_gts, chest_drained_probs))
            chest_drained_accuracy = accuracy_score(chest_drained_gts, chest_drained_preds)
            chest_drained_f1 = f1_score(chest_drained_gts, chest_drained_preds)
            chest_drained_precision = precision_score(chest_drained_gts, chest_drained_preds)
            chest_drained_recall = recall_score(chest_drained_gts, chest_drained_preds)

            metrics["chest_drained/auc"] = aucs[-1]
            metrics["chest_drained/ap"] = aps[-1]
            metrics["chest_drained/acc"] = chest_drained_accuracy
            metrics["chest_drained/f1"] = chest_drained_f1
            metrics["chest_drained/precision"] = chest_drained_precision
            metrics["chest_drained/recall"] = chest_drained_recall

            # chest-w/o-drain metrics
            chest_wo_drain_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx3]])
            chest_wo_drain_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx3][:, 1]])
            chest_wo_drain_preds = (chest_wo_drain_probs >= threshold).astype(int)
            aucs.append(roc_auc_score(chest_wo_drain_gts, chest_wo_drain_probs))
            chest_wo_drain_accuracy = accuracy_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_f1 = f1_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_precision = precision_score(chest_wo_drain_gts, chest_wo_drain_preds)
            chest_wo_drain_recall = recall_score(chest_wo_drain_gts, chest_wo_drain_preds)

            metrics["chest_wo_drained/auc"] = aucs[-1]
            metrics["chest_wo_drained/ap"] = aps[-1]
            metrics["chest_wo_drained/acc"] = chest_wo_drain_accuracy
            metrics["chest_wo_drained/f1"] = chest_wo_drain_f1
            metrics["chest_wo_drained/precision"] = chest_wo_drain_precision
            metrics["chest_wo_drained/recall"] = chest_wo_drain_recall

        else:        
            # For validation set, compute optimal threshold
            if is_validation:
                fpr, tpr, thresholds = roc_curve(gts_numpy, probs_numpy[:, 1])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                threshold = optimal_threshold
                pred_numpy = (probs_numpy[:, 1] >= optimal_threshold).astype(int)
            else:
                # For test set, use provided threshold
                pred_numpy = (probs_numpy[:, 1] >= threshold).astype(int)

            # Create metrics dictionary
            metrics = {}

            # Overall metrics
            aps.append(average_precision_score(gts_numpy, probs_numpy[:, 1]))
            aucs.append(roc_auc_score(gts_numpy, probs_numpy[:, 1]))
            metrics["overall/auc"] = aucs[-1]
            metrics["overall/ap"] = aps[-1]
            metrics["overall/acc"] = accuracy_score(gts_numpy, pred_numpy)
            metrics["overall/f1"] = f1_score(gts_numpy, pred_numpy)
            metrics["overall/precision"] = precision_score(gts_numpy, pred_numpy)
            metrics["overall/recall"] = recall_score(gts_numpy, pred_numpy)

            # Aligned metrics
            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 0))
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 1))
            aligned_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx2]])
            aligned_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])
            aligned_preds = (aligned_probs >= threshold).astype(int)
            
            aps.append(average_precision_score(aligned_gts, aligned_probs))
            aucs.append(roc_auc_score(aligned_gts, aligned_probs))
            metrics["aligned/auc"] = aucs[-1]
            metrics["aligned/ap"] = aps[-1]
            metrics["aligned/acc"] = accuracy_score(aligned_gts, aligned_preds)
            metrics["aligned/f1"] = f1_score(aligned_gts, aligned_preds)
            metrics["aligned/precision"] = precision_score(aligned_gts, aligned_preds)
            metrics["aligned/recall"] = recall_score(aligned_gts, aligned_preds)

            # Conflict metrics
            idx1 = np.where((bias_gts_numpy == 0) & (gts_numpy == 1))
            idx2 = np.where((bias_gts_numpy == 1) & (gts_numpy == 0))
            conflict_gts = np.concatenate([gts_numpy[idx1], gts_numpy[idx2]])
            conflict_probs = np.concatenate([probs_numpy[idx1][:, 1], probs_numpy[idx2][:, 1]])
            conflict_preds = (conflict_probs >= threshold).astype(int)

            aps.append(average_precision_score(conflict_gts, conflict_probs))
            aucs.append(roc_auc_score(conflict_gts, conflict_probs))
            metrics["conflict/auc"] = aucs[-1]
            metrics["conflict/ap"] = aps[-1]
            metrics["conflict/acc"] = accuracy_score(conflict_gts, conflict_preds)
            metrics["conflict/f1"] = f1_score(conflict_gts, conflict_preds)
            metrics["conflict/precision"] = precision_score(conflict_gts, conflict_preds)
            metrics["conflict/recall"] = recall_score(conflict_gts, conflict_preds)

            # balanced metrics
            # Balanced metrics (average of aligned and conflict)
            metrics["balanced/auc"] = (metrics["aligned/auc"] + metrics["conflict/auc"]) / 2
            metrics["balanced/ap"] = (metrics["aligned/ap"] + metrics["conflict/ap"]) / 2
            metrics["balanced/acc"] = (metrics["aligned/acc"] + metrics["conflict/acc"]) / 2
            metrics["balanced/f1"] = (metrics["aligned/f1"] + metrics["conflict/f1"]) / 2
            metrics["balanced/precision"] = (metrics["aligned/precision"] + metrics["conflict/precision"]) / 2
            metrics["balanced/recall"] = (metrics["aligned/recall"] + metrics["conflict/recall"]) / 2
            aps.append((metrics["aligned/ap"] + metrics["conflict/ap"]) / 2)
            aucs.append((metrics["aligned/auc"] + metrics["conflict/auc"]) / 2)

        model.train()
        if is_validation:
            return metrics, optimal_threshold
        else:
            return metrics
    else:
        return accs