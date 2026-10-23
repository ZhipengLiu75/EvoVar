import argparse
import copy
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import torch
from torch.utils.data import Dataset

from utils.Continual import forgetting, average_accuracy, performance_aware_forgetting
from Model import TemporalEncoder
from sklearn.metrics import accuracy_score
from torch import optim
import os

RED = '\033[91m'
RESET = '\033[0m'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'HAR_Var'))

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default=DEFAULT_DATA_PATH, type=str)
parser.add_argument('--tasks', default=10, type=int)
parser.add_argument('--num_class', default=6, type=int)
parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
parser.add_argument('--temporal_channels', type=int, default=32,
                    help='number of local temporal convolution features')
parser.add_argument('--temporal_size', type=int, default=128,
                    help='dimension of each variable representation')
parser.add_argument('--identity_dim', type=int, default=32,
                    help='dimension of the variable identity embedding')
parser.add_argument('--num_slots', type=int, default=8,
                    help='number of fixed latent slots')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--ffn_dim', type=int, default=128,
                    help='hidden dimension of slot feed-forward networks')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout used by slot modules')
parser.add_argument('--kl_weight', type=float, default=0.5,
                    help='weight of variable-subset consistency loss')
parser.add_argument('--kl_temperature', type=float, default=2.0,
                    help='temperature used by the KL consistency loss')
parser.add_argument('--subset_keep_ratio', type=float, default=0.7,
                    help='fraction of current-domain variables retained in the subset view')
parser.add_argument('--history_weight', type=float, default=0.1,
                    help='weight of previous-model temporal feature stabilization')

parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoches', default=150, type=int)
parser.add_argument('--model', type=str, default='TemporalEncoder')


class EarlyStopping:
    def __init__(self, model, patience=15, verbose=False, dump=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = copy.deepcopy(model)

        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())

        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0


class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float().to(args.device)
            self.y_data = torch.from_numpy(y_train).long().to(args.device)
        else:
            self.x_data = X_train.float().to(args.device)
            self.y_data = y_train.long().to(args.device)

        if X_train.shape[1] == args.seq_len:
            num_variables = X_train.shape[2]
        elif X_train.shape[2] == args.seq_len:
            num_variables = X_train.shape[1]
        else:
            raise ValueError(
                f"Cannot find seq_len={args.seq_len} in sample shape {X_train.shape}."
            )

        if "variable_ids" not in dataset:
            raise KeyError(
                "HAR_Var requires a 'variable_ids' field in every domain."
            )
        variable_ids = torch.as_tensor(dataset["variable_ids"], dtype=torch.long)
        if variable_ids.ndim != 1 or variable_ids.numel() != num_variables:
            raise ValueError(
                f"Expected variable_ids with shape [{num_variables}], "
                f"but received {tuple(variable_ids.shape)}."
            )
        self.variable_ids = variable_ids.to(args.device)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.variable_ids

    def __len__(self):
        return self.len


def data_generator(args, task):
    train_datasets = torch.load(os.path.join(args.data_path, "train.pt"))
    val_datasets = torch.load(os.path.join(args.data_path, "val.pt"))
    test_datasets = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = train_datasets[task]
    val_dataset = val_datasets[task]
    test_dataset = test_datasets[task]

    split_ids = []
    for split_name, domain in (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
    ):
        if "variable_ids" not in domain:
            raise KeyError(
                f"Domain {task + 1} of {split_name}.pt has no variable_ids."
            )
        split_ids.append(torch.as_tensor(domain["variable_ids"], dtype=torch.long))
    if not (
            torch.equal(split_ids[0], split_ids[1])
            and torch.equal(split_ids[0], split_ids[2])
    ):
        raise ValueError(
            f"Domain {task + 1} must use identical variable_ids in train/val/test."
        )

    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                                               shuffle=True, drop_last=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batchsize,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader


def sample_variable_subset(data, variable_ids, keep_ratio):
    """Select one non-empty proper variable subset shared by the batch."""
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError("subset_keep_ratio must lie strictly between 0 and 1.")
    if variable_ids.ndim != 2:
        raise ValueError(
            f"Expected batched variable_ids with shape [B, C], got {tuple(variable_ids.shape)}."
        )

    num_variables = variable_ids.size(1)
    if num_variables < 2:
        return data, variable_ids

    keep_count = int(round(num_variables * keep_ratio))
    keep_count = max(1, min(num_variables - 1, keep_count))
    indices = torch.randperm(num_variables, device=data.device)[:keep_count]
    indices = indices.sort().values

    if data.size(1) == num_variables:
        subset_data = data.index_select(1, indices)
    elif data.size(2) == num_variables:
        subset_data = data.index_select(2, indices)
    else:
        raise ValueError(
            "Cannot match the variable dimension between samples and variable_ids: "
            f"samples={tuple(data.shape)}, variable_ids={tuple(variable_ids.shape)}."
        )
    subset_ids = variable_ids.index_select(1, indices)
    return subset_data, subset_ids


def variable_subset_kl(full_logits, subset_logits, temperature):
    if temperature <= 0.0:
        raise ValueError("kl_temperature must be positive.")
    full_probabilities = torch.softmax(
        full_logits.detach() / temperature, dim=-1
    )
    subset_log_probabilities = torch.log_softmax(
        subset_logits / temperature, dim=-1
    )
    return F.kl_div(
        subset_log_probabilities,
        full_probabilities,
        reduction='batchmean',
    ) * (temperature ** 2)


def make_history_teacher(model):
    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)

    teacher.train()
    for module in teacher.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    return teacher


def historical_feature_stability(student_features, teacher_features):
    """Keep the shared temporal representation close to the previous model."""
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Student and teacher temporal features must have identical shapes, "
            f"got {tuple(student_features.shape)} and {tuple(teacher_features.shape)}."
        )
    return F.mse_loss(student_features, teacher_features.detach())


def Test(test_loader, bestmodel, final):
    predicts = None
    labels = None
    bestmodel.eval()
    for data, label, variable_ids in test_loader:
        out = bestmodel(data, variable_ids)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    X = None
    if final == True:
        num_classes = torch.max(labels).item() + 1
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)
        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()
        per_class_acc = correct / total.clamp(min=1)
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item())

    acc = accuracy_score(predicts, labels)
    printtext = "ACC:{:.4f}".format(acc)
    print(printtext)
    return acc, X


def Trainer(args, model):
    PredLossFun = nn.CrossEntropyLoss()
    if args.history_weight < 0.0:
        raise ValueError("history_weight must be non-negative.")
    all_test_loaders = []
    accuracy_matrix = []
    XX = []
    for task in range(args.tasks):
        print('=' * 20, task + 1, '=' * 30)
        history_teacher = None
        if task > 0:
            history_teacher = make_history_teacher(model)

        train_loader, val_loader, test_loader = data_generator(args, task)
        domain_variable_ids = train_loader.dataset.variable_ids.detach().cpu().tolist()
        print(f"Variable IDs: {domain_variable_ids}")
        model.register_variable_ids(train_loader.dataset.variable_ids)
        early_stopping = EarlyStopping(model, patience=15, verbose=True, dump=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)  # , betas=(.9, .99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, patience=3)
        all_test_loaders.append(test_loader)

        for epoch in range(1, args.epoches + 1):
            train_losses, train_cls_losses = [], []
            train_kl_losses, train_history_losses = [], []
            val_losses, test_losses = [], []
            model.train()
            for data, label, variable_ids in train_loader:
                optimizer.zero_grad()
                full_logits, full_aux = model(
                    data,
                    variable_ids,
                    return_aux=True,
                )
                classification_loss = PredLossFun(full_logits, label)

                if history_teacher is None:
                    history_loss = full_logits.new_zeros(())
                else:
                    with torch.no_grad():
                        previous_temporal_features = history_teacher.encode_variables(data)
                    history_loss = historical_feature_stability(
                        full_aux["temporal_features"],
                        previous_temporal_features,
                    )

                subset_data, subset_ids = sample_variable_subset(
                    data,
                    variable_ids,
                    args.subset_keep_ratio,
                )
                subset_logits = model(subset_data, subset_ids)
                kl_loss = variable_subset_kl(
                    full_logits,
                    subset_logits,
                    args.kl_temperature,
                )
                loss = (
                        classification_loss
                        + args.kl_weight * kl_loss
                        + args.history_weight * history_loss
                )

                train_losses.append(loss.item())
                train_cls_losses.append(classification_loss.item())
                train_kl_losses.append(kl_loss.item())
                train_history_losses.append(history_loss.item())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                for data, label, variable_ids in val_loader:
                    out = model(data, variable_ids)
                    loss = PredLossFun(out, label)
                    val_losses.append(loss.item())

            with torch.no_grad():
                model.eval()
                for data, label, variable_ids in test_loader:
                    out = model(data, variable_ids)
                    loss = PredLossFun(out, label)
                    test_losses.append(loss.item())

            print(
                'epoch:{0:}, train_loss:{1:.5f}, cls_loss:{2:.5f}, '
                'kl_loss:{3:.5f}, history_loss:{4:.5f}, '
                'val_loss:{5:.5f}, test_loss:{6:.5f}'.format(
                    epoch,
                    np.mean(train_losses),
                    np.mean(train_cls_losses),
                    np.mean(train_kl_losses),
                    np.mean(train_history_losses),
                    np.mean(val_losses),
                    np.mean(test_losses),
                )
            )

            scheduler.step(np.mean(val_losses))
            early_stopping(np.mean(val_losses), model, epoch)
            if early_stopping.early_stop:
                print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
                break
            if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
                break

        model.load_state_dict(early_stopping.best_model.state_dict())
        accs = []
        for i, test_loader in enumerate(all_test_loaders):
            acc, xx = Test(test_loader, early_stopping.best_model, True)
            print(f"Accuracy on Domain {i + 1}: {acc:.4f}")
            accs.append(acc)
            if task == args.tasks - 1:
                XX.append(xx)
        accuracy_matrix.append(accs)

    print("\n=== Accuracy Matrix ===")
    for i, acc_row in enumerate(accuracy_matrix):
        row_str = ", ".join(f"{acc:.4f}" for acc in acc_row)
        print(f"After Domain {i + 1}: [{row_str}]")

    avg_acc = average_accuracy(accuracy_matrix)
    avg_forget = forgetting(accuracy_matrix)
    avg_pa_forget = performance_aware_forgetting(accuracy_matrix)
    XX = torch.tensor(XX)
    XX = torch.mean(XX, dim=0)
    print(f"\n=== Average Accuracy: {avg_acc:.4f} ===")
    print(f"=== Average Forgetting: {avg_forget:.4f} ===")
    print(f"=== Performance Aware Forgetting: {avg_pa_forget:.4f} ===")

    print(XX.cpu().numpy())
    return avg_acc, avg_forget, avg_pa_forget, XX.cpu().numpy(), accuracy_matrix


def main(args, i=2026):
    SEED = i
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    if args.model == 'TemporalEncoder':
        model = TemporalEncoder(args).float().to(args.device)
    return Trainer(args, model)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_name = os.path.basename(os.path.normpath(args.data_path))


    avg_acc, avg_forget, avg_pa_forget, _, _ = main(args)
    print(avg_acc,avg_forget,avg_pa_forget)
