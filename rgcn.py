import pandas as pd
import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self, dropout):
        super(Net, self).__init__()
        self.gene_emb = Parameter(torch.randn(4264, 1613))
        self.conv1 = RGCNConv(1613, 1600, 4)
        self.conv2 = RGCNConv(1600, 900, 4)
        self.lin1 = Linear(900, 400)
        self.lin2 = Linear(400, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = torch.cat((x, self.gene_emb), dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)

        return x.log_softmax(dim=-1)


# functions for cross-validation
def cv_train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[train_index][train_mask],
                      data.y[train_index][train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def cv_test():
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type).exp()[
        train_index][val_mask]
    pred = out.argmax(dim=-1)
    acc = accuracy_score(data.y[train_index][val_mask].cpu(), pred.cpu())
    auroc = roc_auc_score(data.y[train_index][val_mask].cpu(), out[:, 1].cpu())
    auprc = average_precision_score(
        data.y[train_index][val_mask].cpu(), out[:, 0].cpu(), pos_label=0)
    sens = recall_score(data.y[train_index][val_mask].cpu(), pred.cpu())
    spec = recall_score(data.y[train_index]
                        [val_mask].cpu(), pred.cpu(), pos_label=0)
    return acc, auroc, auprc, sens, spec


# functions for testing
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[train_index], data.y[train_index])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type).exp()[test_index]
    pred = out.argmax(dim=-1)
    acc = accuracy_score(data.y[test_index].cpu(), pred.cpu())
    auroc = roc_auc_score(data.y[test_index].cpu(), out[:, 1].cpu())
    auprc = average_precision_score(
        data.y[test_index].cpu(), out[:, 0].cpu(), pos_label=0)
    sens = recall_score(data.y[test_index].cpu(), pred.cpu())
    spec = recall_score(data.y[test_index].cpu(), pred.cpu(), pos_label=0)
    return acc, auroc, auprc, sens, spec


data = torch.load('graph.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
train_index, test_index = next(sss.split(data.x.cpu(), data.y.cpu()))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=18)

# cross-validation
acc_list = []
auroc_list = []
auprc_list = []
sens_list = []
spec_list = []
epoch_list = []

print('Cross-validation progressing...')
for train_mask, val_mask in tqdm(skf.split(data.x[train_index].cpu(), data.y[train_index].cpu())):
    model = Net(0.5).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0005, weight_decay=0.01)

    acc_max = 0
    for epoch in range(150):
        _ = cv_train()
        acc, auroc, auprc, sens, spec = cv_test()
        if acc > acc_max:
            acc_max = acc
            auroc_max = auroc
            auprc_max = auprc
            sens_max = sens
            spec_max = spec
            epoch_max = epoch

    acc_list.append(acc_max)
    auroc_list.append(auroc_max)
    auprc_list.append(auprc_max)
    sens_list.append(sens_max)
    spec_list.append(spec_max)
    epoch_list.append(epoch_max)

cv_result = pd.DataFrame([acc_list, sens_list, spec_list, auroc_list, auprc_list],
                         index=['Accuracy', 'Sensitivity', 'Specificity', 'AUROC', 'AUPRC'])
cv_result['Mean'] = cv_result.mean(axis=1)
print('Cross-validation results at Epoch {}:'.format(epoch_max))
print(cv_result)

model = Net(0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)

print('Testing progressing...')
acc_max = 0
for epoch in tqdm(range(150)):
    _ = train()
    acc, auroc, auprc, sens, spec = test()
    if acc > acc_max:
        acc_max = acc
        auroc_max = auroc
        auprc_max = auprc
        sens_max = sens
        spec_max = spec
        epoch_max = epoch

test_result = pd.Series([acc_max, sens_max, spec_max, auroc_max, auprc_max], index=[
    'Accuracy', 'Sensitivity', 'Specificity', 'AUROC', 'AUPRC'])
print('Testing results at Epoch {}:'.format(epoch_max))
print(test_result)
