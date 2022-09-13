#!/usr/bin/env python3
from lib.model.resnet_imagenet import resnet50
import torch
from torch.utils.data import DataLoader, random_split
import torch_toolbox
from torch_toolbox.datasets.dataset import (
    get_classification_dataset,
    preprocessing_add_parser_options,
    preprocessing_from_parser,
)
from torch_toolbox.datasets.preprocessing import split_transform
import argparse
from argparse import RawTextHelpFormatter

ap = argparse.ArgumentParser(
    description='Evaluate a FSS Out-of-Distribution detector.',
    formatter_class=RawTextHelpFormatter)
ap.add_argument('-m', '--model', type=str, required=True,
                metavar='<path_to_file>',
                help='Path to trained model.')
preprocessing_add_parser_options(ap)
ap.add_argument('--ind-name', type=str, required=True,
                metavar='<name>',
                help='In-distribution dataset name.')
ap.add_argument('--ind-location', type=str, required=True,
                metavar='<path_to_directory>',
                help='In-distribution dataset location.')
ap.add_argument('--ood-name', type=str, required=True,
                metavar='<name>',
                help='Out-of-distribution dataset name.')
ap.add_argument('--ood-location', type=str, required=True,
                metavar='<path_to_directory>',
                help='Out-of-distribution dataset location.')
ap.add_argument('--val-name', type=str, required=True,
                metavar='<name>',
                help='Validation dataset name.')
ap.add_argument('--val-location', type=str, required=True,
                metavar='<path_to_directory>',
                help='Validation dataset location.')
ap.add_argument('--batch-size', type=int, required=False,
                default=16,
                metavar='<val>',
                help='Batch size.')
ap.add_argument('--device', type=str, required=False,
                metavar='<device_id>',
                default='cpu',
                help='Target device for execution.')
ap.add_argument("--input-process", action='store_true',
                help="Enable input preprocessing")
ap.add_argument('-v', "--verbose", action='store_true',
                help="Verbose mode")
args = ap.parse_args()

# ----- load pre-trained model -----
model = torch_toolbox.load(args.model, map_location='cpu')
if model.__class__.__name__ == 'ResNet':
    # Convert to custom ResNet50
    nclasses = model.fc.out_features
    new_model = resnet50(num_classes=nclasses)
    new_model.load_state_dict(model.state_dict())
    model = new_model.to(args.device)

# ----- load dataset -----
input_ops, target_ops = preprocessing_from_parser(args)
_, reshape, norm = split_transform(input_ops[0])
img_size = reshape.size[0]
std = tuple(norm.transforms[-1].std)
inp_channel = 3  # Always consider RGB images
input_process = args.input_process

ind_trainset = get_classification_dataset(
    name=args.ind_name,
    root=args.ind_location,
    split="train",
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ind_testset = get_classification_dataset(
    name=args.ind_name,
    root=args.ind_location,
    split="test",
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ood_valset = get_classification_dataset(
    name=args.val_name,
    root=args.val_location,
    split=None,
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ood_testset = get_classification_dataset(
    name=args.ood_name,
    root=args.ood_location,
    split="test",
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
# Get 2x 500 random pairs of elements from the (InD trainset, OoD valset)
ind_trainloader, ind_valloader, _ = \
    random_split(ind_trainset, [500, 500, len(ind_trainset) - 1000], generator=torch.Generator().manual_seed(42))
ood_trainloader, ood_valloader, _ = \
    random_split(ood_valset, [500, 500, len(ood_valset) - 1000], generator=torch.Generator().manual_seed(42))
ind_trainloader = DataLoader(ind_trainloader, batch_size=args.batch_size, shuffle=True)
ind_valloader = DataLoader(ind_valloader, batch_size=args.batch_size, shuffle=True)
ood_trainloader = DataLoader(ood_trainloader, batch_size=args.batch_size, shuffle=True)
ood_valloader = DataLoader(ood_valloader, batch_size=args.batch_size, shuffle=True)

ind_testloader = DataLoader(ind_testset, batch_size=args.batch_size, shuffle=True)
ood_testloader = DataLoader(ood_testset, batch_size=args.batch_size, shuffle=True)

from lib.inference import get_feature_dim_list
from lib.inference.FSS import (
    compute_fss,
    get_FSS_score_ensem,
    get_FSS_score_ensem_process,
    search_FSS_hyperparams
)
from lib.metric import get_metrics, train_lr

# ----- Calcualte FSS -----
feature_dim_list, _ = get_feature_dim_list(model, img_size, inp_channel, flat=True)
fss = compute_fss(model, len(feature_dim_list), img_size, inp_channel)
layer_indexs = list(range(len(feature_dim_list)))

# ----- Calculate best magnitude for input pre-processing -----
if input_process:
    best_magnitude = search_FSS_hyperparams(model,
                                            fss,
                                            layer_indexs,
                                            ind_trainloader,
                                            ood_trainloader,
                                            ind_valloader,
                                            ood_valloader,
                                            std=std)

# ----- Calculate FSSD -----
if not input_process:  # when no input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_trainfeats = get_FSS_score_ensem(model, ind_trainloader, fss, layer_indexs)
    print('Get FSSD for OoD validation data.')
    ood_trainfeats = get_FSS_score_ensem(model, ood_trainloader, fss, layer_indexs)

    print('Get FSSD for in-distribution test data.')
    ind_testfeats = get_FSS_score_ensem(model, ind_testloader, fss, layer_indexs)
    print('Get FSSD for OoD test data.')
    ood_testfeats = get_FSS_score_ensem(model, ood_testloader, fss, layer_indexs)
else:  # when input pre-processing is used
    print('Get FSSD for in-distribution validation data.')
    ind_trainfeats = get_FSS_score_ensem_process(model, ind_trainloader, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD validation data.')
    ood_trainfeats = get_FSS_score_ensem_process(model, ood_trainloader, fss, layer_indexs, best_magnitude, std)

    print('Get FSSD for in-distribution test data.')
    ind_testfeats = get_FSS_score_ensem_process(model, ind_testloader, fss, layer_indexs, best_magnitude, std)
    print('Get FSSD for OoD test data.')
    ood_testfeats = get_FSS_score_ensem_process(model, ood_testloader, fss, layer_indexs, best_magnitude, std)

# ----- Training OoD detector using validation data -----
lr = train_lr(ind_trainfeats, ood_trainfeats)

metrics = get_metrics(lr, ind_testfeats, ood_testfeats, acc_type="best")
print("metrics:", metrics)
