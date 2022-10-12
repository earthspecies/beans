import ast
import argparse
import copy
import itertools
import random
import sys
import yaml

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from beans.metrics import Accuracy, MeanAveragePrecision
from beans.models import ResNetClassifier, VGGishClassifier
from beans.datasets import ClassificationDataset, RecognitionDataset


def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets}


def spec2feats(spec):
    spec = torch.cat([
        spec.mean(dim=1),
        spec.std(dim=1),
        spec.min(dim=1)[0],
        spec.max(dim=1)[0]])
    return spec.numpy().reshape(-1)


def eval_sklearn_model(model_and_scaler, dataloader, num_labels, metric_factory):
    total_loss = 0.
    metric = metric_factory()
    model, scaler = model_and_scaler

    for x, y in dataloader:
        xs = [spec2feats(x[i]) for i in range(x.shape[0])]
        xs_scaled = scaler.transform(xs)
        pred = model.predict(xs_scaled)
        if isinstance(model, MultiOutputClassifier):
            pred = torch.tensor(pred)
        else:
            pred = F.one_hot(torch.tensor(pred), num_classes=num_labels)
        metric.update(pred, y)

    return total_loss, metric.get_primary_metric()


def train_sklearn_model(args, dataloader_train, dataloader_valid, num_labels, metric_factory, log_file):
    print(f'Building training data ...', file=sys.stderr)

    xs = []
    ys = []
    for x, y in dataloader_train:
        xs.extend(spec2feats(x[i]) for i in range(x.shape[0]))
        ys.extend(y[i].numpy() for i in range(y.shape[0]))

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs)
    print(f"Num. features = {xs_scaled[0].shape}, num. instances = {len(xs_scaled)}", file=sys.stderr)

    params = ast.literal_eval(args.params)
    assert(isinstance(params, dict))
    param_list = [[(k, v) for v in vs] for k, vs in params.items()]
    param_combinations = itertools.product(*param_list)

    valid_metric_best = 0.
    best_model = None

    for extra_params in param_combinations:
        extra_params = dict(extra_params)
        print(f'Fitting data (params: {extra_params})...', file=sys.stderr)

        if args.model_type == 'lr':
            model = LogisticRegression(max_iter=1_000, **extra_params)
        elif args.model_type == 'svm':
            model = SVC(**extra_params)
        elif args.model_type == 'decisiontree':
            model = DecisionTreeClassifier(**extra_params)
        elif args.model_type == 'gbdt':
            model = GradientBoostingClassifier(**extra_params)
        elif args.model_type == 'xgboost':
            model = XGBClassifier(n_jobs=4, **extra_params)

        if args.task == 'detection':
            model = MultiOutputClassifier(model)

        model.fit(xs_scaled, ys)

        _, valid_metric = eval_sklearn_model(
            model_and_scaler=(model, scaler),
            dataloader=dataloader_valid,
            num_labels=num_labels,
            metric_factory=metric_factory)

        if valid_metric > valid_metric_best:
            best_model = model
            valid_metric_best = valid_metric

        print({
            'extra_params': extra_params,
            'valid': {
                'metric': valid_metric
            }}, file=log_file)

    return (best_model, scaler), valid_metric_best


def eval_pytorch_model(model, dataloader, metric_factory, device, desc):
    model.eval()
    total_loss = 0.
    steps = 0
    metric = metric_factory()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            loss, logits = model(x, y)
            total_loss += loss.cpu().item()
            steps += 1

            metric.update(logits, y)

    total_loss /= steps

    return total_loss, metric.get_primary_metric()


def train_pytorch_model(
    args,
    dataloader_train,
    dataloader_valid,
    num_labels,
    metric_factory,
    sample_rate,
    device,
    log_file):

    lrs = ast.literal_eval(args.lrs)
    assert isinstance(lrs, list)

    valid_metric_best = 0.
    best_model = None

    for lr in lrs:
        print(f"lr = {lr}" , file=log_file)

        if args.model_type.startswith('resnet') and args.task == 'classification':
            pretrained = args.model_type.endswith('pretrained')
            model = ResNetClassifier(
                model_type=args.model_type,
                pretrained=pretrained,
                num_classes=num_labels).to(device)
        elif args.model_type.startswith('resnet') and args.task == 'detection':
            pretrained = args.model_type.endswith('pretrained')
            model = ResNetClassifier(
                model_type=args.model_type,
                pretrained=pretrained,
                num_classes=num_labels,
                multi_label=True).to(device)
        elif args.model_type == 'vggish':
            model = VGGishClassifier(
                sample_rate=sample_rate,
                num_classes=num_labels,
                multi_label=(args.task=='detection')).to(device)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        for epoch in range(args.epochs):
            print(f'epoch = {epoch}', file=sys.stderr)

            model.train()

            train_loss = 0.
            train_steps = 0
            train_metric = metric_factory()

            for x, y in tqdm(dataloader_train, desc='train'):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                loss, logits = model(x, y)

                loss.backward()

                optimizer.step()

                train_loss += loss.cpu()
                train_steps += 1

                train_metric.update(logits, y)

            valid_loss, valid_metric = eval_pytorch_model(
                model=model,
                dataloader=dataloader_valid,
                metric_factory=metric_factory,
                device=device,
                desc='valid')

            if valid_metric > valid_metric_best:
                valid_metric_best = valid_metric
                best_model = copy.deepcopy(model)

            print({
                'epoch': epoch,
                'train': {
                    'loss': (train_loss / train_steps).cpu().item(),
                    'metric': train_metric.get_metric(),
                },
                'valid': {
                    'loss': valid_loss,
                    'metric': valid_metric
                }
            }, file=log_file)
            log_file.flush()

    return best_model, valid_metric_best


def main():
    datasets = read_datasets('datasets.yml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lrs', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--task', choices=['classification', 'detection'])
    parser.add_argument('--model-type', choices=[
        'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost',
        'resnet18', 'resnet18-pretrained',
        'resnet50', 'resnet50-pretrained',
        'resnet152', 'resnet152-pretrained',
        'vggish'])
    parser.add_argument('--dataset', choices=datasets.keys())
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--stop-shuffle', action='store_true')
    parser.add_argument('--log-path', type=str)
    args = parser.parse_args()

    torch.random.manual_seed(42)
    random.seed(42)
    if args.log_path:
        log_file = open(args.log_path, mode='w')
    else:
        log_file = sys.stderr

    device = torch.device('cuda:0')

    if args.model_type == 'vggish':
        feature_type = 'vggish'
    elif args.model_type.startswith('resnet'):
        feature_type = 'melspectrogram'
    else:
        feature_type = 'mfcc'

    dataset = datasets[args.dataset]
    num_labels = dataset['num_labels']

    if dataset['type'] == 'classification':
        dataset_train = ClassificationDataset(
            metadata_path=dataset['train_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)
        dataset_valid = ClassificationDataset(
            metadata_path=dataset['valid_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)
        dataset_test = ClassificationDataset(
            metadata_path=dataset['test_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type=feature_type)

    elif dataset['type'] == 'detection':
        dataset_train = RecognitionDataset(
            metadata_path=dataset['train_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
        dataset_valid = RecognitionDataset(
            metadata_path=dataset['valid_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
        dataset_test = RecognitionDataset(
            metadata_path=dataset['test_data'],
            num_labels=num_labels,
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=60,
            window_width=dataset['window_width'],
            window_shift=dataset['window_shift'],
            feature_type=feature_type)
    else:
        raise ValueError(f"Invalid dataset type: {dataset['type']}")

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=not args.stop_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True)
    if dataset_test is not None:
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True)
    else:
        dataloader_test = None

    if args.task == 'classification':
        Metric = Accuracy
    elif args.task == 'detection':
        Metric = MeanAveragePrecision

    if args.model_type in {'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost'}:
        model_and_scaler, valid_metric_best = train_sklearn_model(
            args=args,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            num_labels=num_labels,
            metric_factory=Metric,
            log_file=log_file)

        if dataloader_test is not None:
            _, test_metric = eval_sklearn_model(
                model_and_scaler=model_and_scaler,
                dataloader=dataloader_test,
                num_labels=num_labels,
                metric_factory=Metric)

    else:
        model, valid_metric_best = train_pytorch_model(
            args=args,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            num_labels=num_labels,
            metric_factory=Metric,
            sample_rate=dataset.get('sample_rate', 16000),
            device=device,
            log_file=log_file)

        if dataloader_test is not None:
            _, test_metric = eval_pytorch_model(
                model=model,
                dataloader=dataloader_test,
                metric_factory=Metric,
                device=device,
                desc='test')

    print(
        'valid_metric_best = ', valid_metric_best,
        'test_metric = ', test_metric,
        file=log_file)

    if args.log_path:
        log_file.close()

if __name__ == '__main__':
    main()
