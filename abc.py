import os
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from data.util import get_dataset, IdxDataset
from module.util import get_model
from module.loss import GeneralizedCELoss
import argparse
from evaluate import evaluate_acc_ap_auc, evaluate_acc_ap_auc_mhead

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='lff', help="name of this experiment")
    parser.add_argument('--bias_mlp_classifier', action='store_true', help='Define biased network classifier.')
    parser.add_argument('--debias_mlp_classifier', action='store_true', help='Define debiased network classifier.')
    parser.add_argument('--model_tag', type=str, default='ResNet18', help="DenseNet121, ResNet18")
    parser.add_argument('--xrv_weight', action='store_true', help='Whether using densenet 121 prtrained weight from xrv.')
    parser.add_argument('--imagenet_weight', action='store_true', help='Whether using densenet 121 prtrained weight from xrv.')
    parser.add_argument('--data_dir', type=str, default='./dataset', help="Address where you store the csv files.")
    parser.add_argument('--debug', action='store_true', help='False for saving result or Ture for not saving result.')
    parser.add_argument('--dataset_tag', type=str, default='Source_pneumonia_bias90', help="Source_pneumonia_bias90, Source_pneumonia_bias95, \
                                                                                            Source_pneumonia_bias99, Gender_pneumothorax_case1,\
                                                                                            Gender_pneumothorax_case2, Skin_90")    
    parser.add_argument('--log_dir', type=str, default='/jhcnas4/luoluyang/new_log', help="Address to store the log files.")
    parser.add_argument('--seed', type=int, default=1, help="1, 2, 3, or any other seeds")
    parser.add_argument('--device', type=int, default=0, help="0, 1")
    parser.add_argument('--target_attr_idx', type=int, default=0, help="0, 1")
    parser.add_argument('--bias_attr_idx', type=int, default=1, help="0, 1")
    parser.add_argument('--main_batch_size', type=int, default=256)
    parser.add_argument('--main_learning_rate', type=int, default=1e-4)
    parser.add_argument('--main_weight_decay', type=int, default=5e-4)
    parser.add_argument('--main_num_epochs', type=int, default=20, help = "About 25 epochs")
    parser.add_argument('--lamda', type=float, default=300., help = "Loss weight for disagree.")
    parser.add_argument('--num_heads', type=int, default=8, help = "numbers of classifiers.")
    parser.add_argument('--temperature', type=float, default=1, help = "a hyperperameters.")
    parser.add_argument('--proportion', type=float, default=1, help = "a hyperperameters.")

    args = parser.parse_args()
    print(args)
    dataset_tag = args.dataset_tag 
    main_tag = args.dataset_tag + "_seed_{}".format(args.seed)
    log_dir = os.path.join(args.log_dir, args.dataset_tag)

    seed = args.seed
    temperature = args.temperature
    proportion = args.proportion
    num_heads = args.num_heads

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    exp_name = args.exp_name + '_lamda_{}'.format(args.lamda) + '_proportion_{}'.format(proportion) + '_{}_heads'.format(num_heads)

    if "Drain" in dataset_tag:
        is_drain = True
    else:
        is_drain = False
    
    if not args.debug:
        print('saving the result.')
        
        writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag, exp_name))

    print('Device: {}'.format(args.device))
    print('Experiment: {}'.format(exp_name))
    print('Seed: {}'.format(seed))

    train_dataset = get_dataset(
        dataset_tag,
        dataset_split="train",
        transform_split="train",
        num_heads=args.num_heads,
        proportion=args.proportion
    )
    test_dataset = get_dataset(
        dataset_tag,
        dataset_split="test",
        transform_split="test",
        num_heads=args.num_heads,
        proportion=args.proportion
    )
    
    train_target_attr = train_dataset.attr[:, args.target_attr_idx]
    train_bias_attr = train_dataset.attr[:, args.bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]
    
    train_dataset = IdxDataset(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.main_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    test_dataset = IdxDataset(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.main_batch_size*2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(dataset_tag)
    valid_dataset = get_dataset(
        dataset_tag,
        dataset_split="valid",
        transform_split="valid",
        num_heads=args.num_heads,
        proportion=args.proportion
        )
    valid_dataset = IdxDataset(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.main_batch_size*2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # define model and optimizer
    print('num_classes: ', num_classes)
    print('Model: {}'.format(args.model_tag))
    print('Using xrv weight: {}'.format(args.xrv_weight))
    print('Using imagenet weight: {}'.format(args.imagenet_weight))
    model_b = get_model(args.model_tag, num_classes, mlp_classifier=args.bias_mlp_classifier,\
                        pretrained_imagenet=args.imagenet_weight, pretrained_xrv=args.xrv_weight, num_heads=args.num_heads).to(device)
    model_d = get_model(args.model_tag, num_classes, mlp_classifier=args.debias_mlp_classifier,\
                        pretrained_imagenet=args.imagenet_weight, pretrained_xrv=args.xrv_weight, num_heads=None).to(device)
    optimizer_b = torch.optim.Adam(
        model_b.parameters(),
        lr=args.main_learning_rate,
        weight_decay=args.main_weight_decay,
        betas=(0.9, 0.999)
    )
    
    optimizer_d = torch.optim.Adam(
        model_d.parameters(),
        lr=args.main_learning_rate,
        weight_decay=args.main_weight_decay,
        betas=(0.9, 0.999)
    )

    # define loss
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss(reduction='none')
 
    valid_attrwise_accs_list = []

    if "Drain" in dataset_tag:
        val_names=['valid', 'valid_chest_drained', 'valid_chest_wo_drained']
        test_names=['test', 'test_chest_drained', 'test_chest_wo_drained']
        best_auc = dict(zip(val_names, [0.0]*3))
        best_epoch = 0
        test_in_best=dict(zip(test_names, [0.0]*3))
    else:
        val_names=['valid', 'valid_aligned', 'valid_conflict', 'valid_balanced']
        test_names=['test', 'test_aligned', 'test_conflict', 'test_balanced']
        best_auc = dict(zip(val_names, [0.0]*4))
        best_epoch = 0
        test_in_best=dict(zip(test_names, [0.0]*4))

    # train    
    for epoch in range(args.main_num_epochs):
        for iter, (index, path, data, attr, mask) in tqdm(enumerate(train_loader)):
            for i in range(len(mask)):
                mask[i] = mask[i].to(device)

            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, args.target_attr_idx]
            bias_label = attr[:, args.bias_attr_idx]
            if args.model_tag == 'DenseNet121' and args.xrv_weight:
                features_b = model_b.features2(data).to(device)
            elif args.model_tag == 'DenseNet121':
                features_b = model_b.features(data).to(device)
                features_b = nn.AdaptiveAvgPool2d((1, 1))(features_b)
                features_b = torch.flatten(features_b, 1)
            elif args.model_tag == 'ResNet18':
                features_b = model_b.features(data).to(device)
                features_b = features_b.view(features_b.size(0), -1).to(device)
            outputs = []
            loss_b = 0
            prob_b_lst = []

            for i in range(num_heads):

                logit_b = model_b.classifiers[i](features_b)
                prob_b = torch.softmax(logit_b, dim=1)
                prob_b = torch.gather(prob_b, 1, torch.unsqueeze(label, 1))
                prob_b_lst.append(prob_b)

                outputs.append(logit_b * temperature)
                count = mask[i].sum()
                if count > 0:
                    loss_b += ((bias_criterion(outputs[i], label) * mask[i]).to(device)).sum() / mask[i].sum()
            prob_b_g = torch.mean(torch.stack(prob_b_lst), dim=0).detach()
            loss_weight_d = prob_b_g.detach()
            logit_d = model_d(data)

            prob_d = torch.softmax(logit_d, dim=1)
            prob_d_g = torch.gather(prob_d, 1, torch.unsqueeze(label, 1))

            adv_loss_h = ((- torch.log(prob_d_g * (1-prob_b_g.detach()) + prob_b_g.detach() * (1-prob_d_g) +  1e-7))*(1-prob_b_g.detach())).mean()

            loss_d_update = criterion(logit_d, label) * loss_weight_d + args.lamda * adv_loss_h
            loss_d_update = sum(loss_d_update) / len(loss_d_update)
            
            loss = loss_b + loss_d_update.mean() 
            optimizer_b.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_b.step()
            optimizer_d.step()

        # log
        if not args.debug:
            if "Drain" in dataset_tag:
                metrics, optimal_threshold = evaluate_acc_ap_auc(model_d, valid_loader, attr_dims, args, drain=is_drain, is_validation=True)
                valid_attrwise_aucs = [metrics["overall/auc"], metrics["chest_drained/auc"], metrics["chest_wo_drained/auc"]]
                valid_attrwise_f1 = [metrics["overall/f1"], metrics["chest_drained/f1"], metrics["chest_wo_drained/f1"]]
                valid_attrwise_precision = [metrics["overall/precision"], metrics["chest_drained/precision"], metrics["chest_wo_drained/precision"]]
                valid_attrwise_recall = [metrics["overall/recall"], metrics["chest_drained/recall"], metrics["chest_wo_drained/recall"]]


                valid_accs = metrics["overall/acc"]

                writer.add_scalar("acc/valid", valid_accs, epoch)

                writer.add_scalar(
                    "acc/valid_aligned",
                    metrics["chest_drained/acc"],
                    epoch
                )
                writer.add_scalar(
                    "acc/valid_conflict",
                    metrics["chest_wo_drained/acc"],
                    epoch
                )
                
                writer.add_scalar("auc/valid", valid_attrwise_aucs[0], epoch)
                writer.add_scalar("auc/valid_chest_drained", valid_attrwise_aucs[1], epoch)
                writer.add_scalar("auc/valid_chest-w/o-drained", valid_attrwise_aucs[2], epoch)

                writer.add_scalar("f1/valid", valid_attrwise_f1[0], epoch)
                writer.add_scalar("f1/valid_chest_drained", valid_attrwise_f1[1], epoch)
                writer.add_scalar("f1/valid_chest-w/o-drained", valid_attrwise_f1[2], epoch)

                writer.add_scalar("precision/valid", valid_attrwise_precision[0], epoch)
                writer.add_scalar("precision/valid_chest_drained", valid_attrwise_precision[1], epoch)
                writer.add_scalar("precision/valid_chest-w/o-drained", valid_attrwise_precision[2], epoch)

                writer.add_scalar("recall/valid", valid_attrwise_recall[0], epoch)
                writer.add_scalar("recall/valid_chest_drained", valid_attrwise_recall[1], epoch)
                writer.add_scalar("recall/valid_chest-w/o-drained", valid_attrwise_recall[2], epoch)

                # test
                metrics = evaluate_acc_ap_auc(model_d, test_loader, attr_dims, args, drain=is_drain, is_validation=False, threshold=optimal_threshold)
                test_attrwise_aucs = [metrics["overall/auc"], metrics["chest_drained/auc"], metrics["chest_wo_drained/auc"]]
                test_attrwise_f1 = [metrics["overall/f1"], metrics["chest_drained/f1"], metrics["chest_wo_drained/f1"]]
                test_attrwise_precision = [metrics["overall/precision"], metrics["chest_drained/precision"], metrics["chest_wo_drained/precision"]]
                test_attrwise_recall = [metrics["overall/recall"], metrics["chest_drained/recall"], metrics["chest_wo_drained/recall"]]

                test_accs = metrics["overall/acc"]
                writer.add_scalar("acc/test", test_accs, epoch)

                writer.add_scalar(
                    "acc/test_aligned",
                    metrics["chest_drained/acc"],
                    epoch
                )
                writer.add_scalar(
                    "acc/test_conflict",
                    metrics["chest_wo_drained/acc"],
                    epoch
                )


                writer.add_scalar("auc/test", test_attrwise_aucs[0], epoch)
                writer.add_scalar("auc/test_chest_drained", test_attrwise_aucs[1], epoch)
                writer.add_scalar("auc/test_chest-w/o-drained", test_attrwise_aucs[2], epoch)

                writer.add_scalar("f1/test", test_attrwise_f1[0], epoch)
                writer.add_scalar("f1/test_chest_drained", test_attrwise_f1[1], epoch)
                writer.add_scalar("f1/test_chest-w/o-drained", test_attrwise_f1[2], epoch)

                writer.add_scalar("precision/test", test_attrwise_precision[0], epoch)
                writer.add_scalar("precision/test_chest_drained", test_attrwise_precision[1], epoch)
                writer.add_scalar("precision/test_chest-w/o-drained", test_attrwise_precision[2], epoch)

                writer.add_scalar("recall/test", test_attrwise_recall[0], epoch)
                writer.add_scalar("recall/test_chest_drained", test_attrwise_recall[1], epoch)
                writer.add_scalar("recall/test_chest-w/o-drained", test_attrwise_recall[2], epoch)

                if valid_attrwise_aucs[0]>list(best_auc.values())[0]:
                    best_auc = dict(zip(val_names, valid_attrwise_aucs))
                    best_epoch = epoch
                    test_in_best = dict(zip(test_names, test_attrwise_aucs))
                    state_dict = {
                    'epochs': epoch,
                    'state_dict': model_d.state_dict(),
                    'optimizer': optimizer_d.state_dict()
                    }
            else:
                metrics_b, optimal_threshold_b = evaluate_acc_ap_auc_mhead(model_b, valid_loader, attr_dims, args, drain=is_drain, is_validation=True)
                metrics_d, optimal_threshold_d = evaluate_acc_ap_auc(model_d, valid_loader, attr_dims, args, drain=is_drain, is_validation=True)
                valid_attrwise_aucs_b = [metrics_b["overall/auc"], metrics_b["aligned/auc"], metrics_b["conflict/auc"], metrics_b["balanced/auc"]]
                valid_attrwise_f1_b = [metrics_b["overall/f1"], metrics_b["aligned/f1"], metrics_b["conflict/f1"], metrics_b["balanced/f1"]]
                valid_attrwise_precision_b = [metrics_b["overall/precision"], metrics_b["aligned/precision"], metrics_b["conflict/precision"], metrics_b["balanced/precision"]]
                valid_attrwise_recall_b = [metrics_b["overall/recall"], metrics_b["aligned/recall"], metrics_b["conflict/recall"], metrics_b["balanced/recall"]]

                valid_attrwise_aucs_d = [metrics_d["overall/auc"], metrics_d["aligned/auc"], metrics_d["conflict/auc"], metrics_d["balanced/auc"]]
                valid_attrwise_accs_d = metrics_d["overall/acc"]
                valid_attrwise_f1_d = [metrics_d["overall/f1"], metrics_d["aligned/f1"], metrics_d["conflict/f1"], metrics_d["balanced/f1"]]
                valid_attrwise_precision_d = [metrics_d["overall/precision"], metrics_d["aligned/precision"], metrics_d["conflict/precision"], metrics_d["balanced/precision"]]
                valid_attrwise_recall_d = [metrics_d["overall/recall"], metrics_d["aligned/recall"], metrics_d["conflict/recall"], metrics_d["balanced/recall"]]


                writer.add_scalar("auc/b_valid", valid_attrwise_aucs_b[0], epoch)
                writer.add_scalar("auc/b_valid_aligned", valid_attrwise_aucs_b[1], epoch)
                writer.add_scalar("auc/b_valid_conflict", valid_attrwise_aucs_b[2], epoch)
                writer.add_scalar("auc/b_valid_balanced", valid_attrwise_aucs_b[3], epoch)

                writer.add_scalar("f1/b_valid", valid_attrwise_f1_b[0], epoch)
                writer.add_scalar("f1/b_valid_aligned", valid_attrwise_f1_b[1], epoch)
                writer.add_scalar("f1/b_valid_conflict", valid_attrwise_f1_b[2], epoch)
                writer.add_scalar("f1/b_valid_balanced", valid_attrwise_f1_b[3], epoch)

                writer.add_scalar("precision/b_valid", valid_attrwise_precision_b[0], epoch)
                writer.add_scalar("precision/b_valid_aligned", valid_attrwise_precision_b[1], epoch)
                writer.add_scalar("precision/b_valid_conflict", valid_attrwise_precision_b[2], epoch)
                writer.add_scalar("precision/b_valid_balanced", valid_attrwise_precision_b[3], epoch)

                writer.add_scalar("recall/b_valid", valid_attrwise_recall_b[0], epoch)
                writer.add_scalar("recall/b_valid_aligned", valid_attrwise_recall_b[1], epoch)
                writer.add_scalar("recall/b_valid_conflict", valid_attrwise_recall_b[2], epoch)
                writer.add_scalar("recall/b_valid_balanced", valid_attrwise_recall_b[3], epoch)
    

                writer.add_scalar("auc/d_valid", valid_attrwise_aucs_d[0], epoch)
                writer.add_scalar("auc/d_valid_aligned", valid_attrwise_aucs_d[1], epoch)
                writer.add_scalar("auc/d_valid_conflict", valid_attrwise_aucs_d[2], epoch)
                writer.add_scalar("auc/d_valid_balanced", valid_attrwise_aucs_d[3], epoch)

                writer.add_scalar("f1/d_valid", valid_attrwise_f1_d[0], epoch)
                writer.add_scalar("f1/d_valid_aligned", valid_attrwise_f1_d[1], epoch)
                writer.add_scalar("f1/d_valid_conflict", valid_attrwise_f1_d[2], epoch)
                writer.add_scalar("f1/d_valid_balanced", valid_attrwise_f1_d[3], epoch)

                writer.add_scalar("precision/d_valid", valid_attrwise_precision_d[0], epoch)
                writer.add_scalar("precision/d_valid_aligned", valid_attrwise_precision_d[1], epoch)
                writer.add_scalar("precision/d_valid_conflict", valid_attrwise_precision_d[2], epoch)
                writer.add_scalar("precision/d_valid_balanced", valid_attrwise_precision_d[3], epoch)

                writer.add_scalar("recall/d_valid", valid_attrwise_recall_d[0], epoch)
                writer.add_scalar("recall/d_valid_aligned", valid_attrwise_recall_d[1], epoch)
                writer.add_scalar("recall/d_valid_conflict", valid_attrwise_recall_d[2], epoch)
                writer.add_scalar("recall/d_valid_balanced", valid_attrwise_recall_d[3], epoch)


                # test
                metrics_b = evaluate_acc_ap_auc_mhead(model_b, test_loader, attr_dims, args, drain=is_drain, is_validation=False, threshold=optimal_threshold_b)
                metrics_d = evaluate_acc_ap_auc(model_d, test_loader, attr_dims, args, drain=is_drain, is_validation=False, threshold=optimal_threshold_d)
                test_attrwise_aucs_b = [metrics_b["overall/auc"], metrics_b["aligned/auc"], metrics_b["conflict/auc"], metrics_b["balanced/auc"]]
                test_attrwise_f1_b = [metrics_b["overall/f1"], metrics_b["aligned/f1"], metrics_b["conflict/f1"], metrics_b["balanced/f1"]]
                test_attrwise_precision_b = [metrics_b["overall/precision"], metrics_b["aligned/precision"], metrics_b["conflict/precision"], metrics_b["balanced/precision"]]
                test_attrwise_recall_b = [metrics_b["overall/recall"], metrics_b["aligned/recall"], metrics_b["conflict/recall"], metrics_b["balanced/recall"]]

                test_attrwise_aucs_d = [metrics_d["overall/auc"], metrics_d["aligned/auc"], metrics_d["conflict/auc"], metrics_d["balanced/auc"]]
                test_attrwise_f1_d = [metrics_d["overall/f1"], metrics_d["aligned/f1"], metrics_d["conflict/f1"], metrics_d["balanced/f1"]]
                test_attrwise_precision_d = [metrics_d["overall/precision"], metrics_d["aligned/precision"], metrics_d["conflict/precision"], metrics_d["balanced/precision"]]
                test_attrwise_recall_d = [metrics_d["overall/recall"], metrics_d["aligned/recall"], metrics_d["conflict/recall"], metrics_d["balanced/recall"]]


                writer.add_scalar("auc/b_test", test_attrwise_aucs_b[0], epoch)
                writer.add_scalar("auc/b_test_aligned", test_attrwise_aucs_b[1], epoch)
                writer.add_scalar("auc/b_test_conflict", test_attrwise_aucs_b[2], epoch)
                writer.add_scalar("auc/b_test_balanced", test_attrwise_aucs_b[3], epoch)

                writer.add_scalar("f1/b_test", test_attrwise_f1_b[0], epoch)
                writer.add_scalar("f1/b_test_aligned", test_attrwise_f1_b[1], epoch)
                writer.add_scalar("f1/b_test_conflict", test_attrwise_f1_b[2], epoch)
                writer.add_scalar("f1/b_test_balanced", test_attrwise_f1_b[3], epoch)

                writer.add_scalar("precision/b_test", test_attrwise_precision_b[0], epoch)
                writer.add_scalar("precision/b_test_aligned", test_attrwise_precision_b[1], epoch)
                writer.add_scalar("precision/b_test_conflict", test_attrwise_precision_b[2], epoch)
                writer.add_scalar("precision/b_test_balanced", test_attrwise_precision_b[3], epoch)

                writer.add_scalar("recall/b_test", test_attrwise_recall_b[0], epoch)
                writer.add_scalar("recall/b_test_aligned", test_attrwise_recall_b[1], epoch)
                writer.add_scalar("recall/b_test_conflict", test_attrwise_recall_b[2], epoch)
                writer.add_scalar("recall/b_test_balanced", test_attrwise_recall_b[3], epoch)

                writer.add_scalar("auc/d_test", test_attrwise_aucs_d[0], epoch)
                writer.add_scalar("auc/d_test_aligned", test_attrwise_aucs_d[1], epoch)
                writer.add_scalar("auc/d_test_conflict", test_attrwise_aucs_d[2], epoch)
                writer.add_scalar("auc/d_test_balanced", test_attrwise_aucs_d[3], epoch)

                writer.add_scalar("f1/d_test", test_attrwise_f1_d[0], epoch)
                writer.add_scalar("f1/d_test_aligned", test_attrwise_f1_d[1], epoch)
                writer.add_scalar("f1/d_test_conflict", test_attrwise_f1_d[2], epoch)
                writer.add_scalar("f1/d_test_balanced", test_attrwise_f1_d[3], epoch)

                writer.add_scalar("precision/d_test", test_attrwise_precision_d[0], epoch)
                writer.add_scalar("precision/d_test_aligned", test_attrwise_precision_d[1], epoch)
                writer.add_scalar("precision/d_test_conflict", test_attrwise_precision_d[2], epoch)
                writer.add_scalar("precision/d_test_balanced", test_attrwise_precision_d[3], epoch)

                writer.add_scalar("recall/d_test", test_attrwise_recall_d[0], epoch)
                writer.add_scalar("recall/d_test_aligned", test_attrwise_recall_d[1], epoch)
                writer.add_scalar("recall/d_test_conflict", test_attrwise_recall_d[2], epoch)
                writer.add_scalar("recall/d_test_balanced", test_attrwise_recall_d[3], epoch)
        
                test_accs_b = metrics_b["overall/acc"]
                test_accs_d = metrics_d["overall/acc"]
                writer.add_scalar("acc/b_test", test_accs_b, epoch)
                writer.add_scalar("acc/d_test", test_accs_d, epoch)

                writer.add_scalar(
                    "acc/b_test_aligned",
                    metrics_b["aligned/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/b_test_conflict",
                    metrics_b["conflict/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/b_test_balanced",
                    metrics_b["balanced/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_test_aligned",
                    metrics_d["aligned/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_test_conflict",
                    metrics_d["conflict/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_test_balanced",
                    metrics_d["balanced/acc"],
                    epoch,
                )


                valid_attrwise_accs_list.append(valid_attrwise_accs_d)

                valid_accs_b = metrics_b["overall/acc"]
                valid_accs_d = metrics_d["overall/acc"]
                writer.add_scalar("acc/b_valid", valid_accs_b, epoch)
                writer.add_scalar("acc/d_valid", valid_accs_d, epoch)

                writer.add_scalar(
                    "acc/b_valid_aligned",
                    metrics_b["aligned/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/b_valid_conflict",
                    metrics_b["conflict/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/b_valid_balanced",
                    metrics_b["balanced/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_valid_aligned",
                    metrics_d["aligned/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_valid_conflict",
                    metrics_d["conflict/acc"],
                    epoch,
                )
                writer.add_scalar(
                    "acc/d_valid_balanced",
                    metrics_d["balanced/acc"],
                    epoch,
                )
                if list(best_auc.values())[0]<valid_attrwise_aucs_d[0]:
                    best_auc = dict(zip(val_names[:-1], valid_attrwise_aucs_d))
                    best_auc.update(list(zip(val_names[-1],[(valid_attrwise_aucs_d[1]+valid_attrwise_aucs_d[2]) / 2]))) 
                    test_in_best = dict(zip(test_names[:-1], test_attrwise_aucs_d))
                    test_in_best.update(list(zip(test_names[-1],[(test_attrwise_aucs_d[1]+test_attrwise_aucs_d[2]) / 2]))) 
                    best_epoch = epoch
                    state_dict = {
                    'epochs': epoch,
                    'state_dict': model_d.state_dict(),
                    'optimizer': optimizer_d.state_dict()
                    }


    print("best_auc: ", best_auc)
    print("corresponding_test_auc: ", test_in_best)
    print("best_epoch: ", best_epoch)

    # save the model
    if not args.debug:
        os.makedirs(os.path.join(log_dir, "model", main_tag), exist_ok=True)
        model_path = os.path.join(log_dir, "model", main_tag, exp_name + ".th")

        with open(model_path, "wb") as f:
            torch.save(state_dict, f)


if __name__ == '__main__':
    train()


