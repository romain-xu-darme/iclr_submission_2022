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

ind_train_dataset = get_classification_dataset(
    name=args.ind_name,
    root=args.ind_location,
    split="train",
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ind_test_dataset = get_classification_dataset(
    name=args.ind_name,
    root=args.ind_location,
    split="test",
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
val_test_dataset = get_classification_dataset(
    name=args.val_name,
    root=args.val_location,
    split=None,
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ood_test_dataset = get_classification_dataset(
    name=args.ood_name,
    root=args.ood_location,
    split=None,
    transform=input_ops[0],
    target_transform=target_ops,
    download=True,
    verbose=True,
)
ind_train_loader = DataLoader(ind_train_dataset, batch_size=args.batch_size, shuffle=True)
ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = \
    random_split(ind_test_dataset, [500, 500, len(ind_test_dataset)-1000], generator=torch.Generator().manual_seed(42))
ind_dataloader_val_for_train = DataLoader(ind_dataloader_val_for_train, batch_size=args.batch_size, shuffle=True)
ind_dataloader_val_for_test = DataLoader(ind_dataloader_val_for_test, batch_size=args.batch_size, shuffle=True)
ind_dataloader_test = DataLoader(ind_dataloader_test, batch_size=args.batch_size, shuffle=True)
ood_dataloader_val_for_train, ood_dataloader_val_for_test, _ = \
    random_split(val_test_dataset, [500, 500, len(val_test_dataset)-1000], generator=torch.Generator().manual_seed(42))
ood_dataloader_val_for_train = DataLoader(ood_dataloader_val_for_train, batch_size=args.batch_size, shuffle=True)
ood_dataloader_val_for_test = DataLoader(ood_dataloader_val_for_test, batch_size=args.batch_size, shuffle=True)

ood_dataloader_test = DataLoader(ood_test_dataset, batch_size=args.batch_size, shuffle=True)

# ind_train_loader = get_dataloader(args['ind'], transform, "train",dataroot=args['dataroot'],batch_size=batch_size)
# ind_test_loader = get_dataloader(args['ind'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
# ood_test_loader = get_dataloader(args['ood'], transform, "test", dataroot=args['dataroot'], batch_size=batch_size)
# ind_dataloader_val_for_train, ind_dataloader_val_for_test, ind_dataloader_test = split_dataloader(args['ind'], ind_test_loader, [500, 500, -1], random=True)
# ood_dataloader_val_for_train, ood_dataloader_val_for_test, ood_dataloader_test = split_dataloader(args['ood'], ood_test_loader, [500,500, -1], random=True)


# if args['ind'] == 'dogs50B':
#     ind_train_loader = get_dataloader('dogs50A', transform, "train",dataroot=args['dataroot'],batch_size=args['batch_size'])

# ---- Calculating Mahanalobis distance ----
from lib.inference import get_feature_dim_list
from lib.inference.Mahalanobis import (
        sample_estimator,
        search_Mahalanobis_hyperparams,
        get_Mahalanobis_score_ensemble,
    )
from lib.metric import get_metrics, train_lr
from lib.utils import split_dataloader

feature_dim_list, num_classes = get_feature_dim_list(model, img_size, inp_channel, flat=False)
# print('number of classes', num_classes)
# print(feature_dim_list)
sample_mean, precision = sample_estimator(model, num_classes, feature_dim_list, ind_train_loader)

layer_indexs = list(range(len(feature_dim_list)))
best_magnitude = 0.005

# best_magnitude = search_Mahalanobis_hyperparams(model, sample_mean, precision, layer_indexs, num_classes,
#                                 ind_dataloader_val_for_train,
#                                 ood_dataloader_val_for_train,
#                                 ind_dataloader_val_for_test,
#                                 ood_dataloader_val_for_test,
#                                 std=std)

#ind_features_val_for_train = get_Mahalanobis_score_ensemble(model, ind_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)
#ood_features_val_for_train = get_Mahalanobis_score_ensemble(model, ood_dataloader_val_for_train, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)

ind_features_test = get_Mahalanobis_score_ensemble(model, ind_dataloader_test, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)
print(ind_features_test.shape)
print(ind_features_test[0])
ood_features_test = get_Mahalanobis_score_ensemble(model, ood_dataloader_test, layer_indexs, num_classes, sample_mean, precision, best_magnitude, std=std)[:len(ind_features_test)]
print(ood_features_test.shape)
# ----- Training OoD detector using validation data -----
#lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)

# ----- Calculating metrics using test data -----
metrics = get_metrics(lr, ind_features_test, ood_features_test, acc_type="best")
print("best params: ", best_magnitude)
print("metrics: ", metrics)

