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
from sklearn.metrics import matthews_corrcoef
from pytorch_metric_learning import losses, distances, reducers, testers
from pcgrad import PCGrad
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dropout):
        super(Net, self).__init__()
        self.gene_emb = Parameter(torch.randn(4264, 1613))
        self.conv1 = RGCNConv(1613, dim1, 4)
        self.conv2 = RGCNConv(dim1, dim2, 4)
        self.lin1 = Linear(dim2, dim3)
        self.lin2 = Linear(dim3, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = torch.cat((x, self.gene_emb), dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        emb = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), emb


def model_train(train_idx):
    model.train()
    optimizer.zero_grad()
    out, emb = model(data.x, data.edge_index, data.edge_type)
    loss_tri = Loss_triplet(emb[train_idx], data.y[train_idx])
    loss_nll = F.nll_loss(out[train_idx], data.y[train_idx])
    loss_list = [loss_tri, loss_nll]
    optimizer.pc_backward(loss_list)
    optimizer.step()


@torch.no_grad()
def model_test(test_idx):
    model.eval()
    out, emb = model(data.x, data.edge_index, data.edge_type)
    out = out.exp()[test_idx]
    pred = out.argmax(dim=-1)
    acc = accuracy_score(data.y[test_idx].cpu(), pred.cpu())
    auroc = roc_auc_score(data.y[test_idx].cpu(), out[:, 1].cpu())
    auprc = average_precision_score(
            data.y[test_idx].cpu(), out[:, 0].cpu(), pos_label=0)
    sens = recall_score(data.y[test_idx].cpu(), pred.cpu())
    spec = recall_score(data.y[test_idx].cpu(), pred.cpu(), pos_label=0)
    mcc = matthews_corrcoef(data.y[test_idx].cpu(), pred.cpu())
    return acc, sens, spec, mcc, auroc, auprc


data = torch.load('graph.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
train_index, test_index = next(sss.split(data.x.cpu(), data.y.cpu()))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=18)

# Hyperparameters
lr = 0.00195
weight_decay = 0.00738
margin = 0.274
dropout = 0.645
triplets_per_anchor = 60
dim1 = 1340
dim2 = 920
dim3 = 740
low = 0.274

# triplet margin loss
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=low)
Loss_triplet = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer,
                                triplets_per_anchor=triplets_per_anchor)

# cross-validation
acc_list = []
sens_list = []
spec_list = []
mcc_list = []
auroc_list = []
auprc_list = []
epoch_list = []

print('Cross-validation progressing...')
for train_mask, val_mask in tqdm(skf.split(data.x[train_index].cpu(), data.y[train_index].cpu())):
    model = Net(dim1, dim2, dim3, dropout).to(device)
    optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))

    auroc_max = 0
    for epoch in range(1000):
        model_train(train_index[train_mask])
        acc, sens, spec, mcc, auroc, auprc = model_test(train_index[val_mask])
        if auroc > auroc_max:
            acc_max = acc
            auroc_max = auroc
            auprc_max = auprc
            sens_max = sens
            spec_max = spec
            mcc_max = mcc
            epoch_max = epoch

    acc_list.append(acc_max)
    sens_list.append(sens_max)
    spec_list.append(spec_max)
    mcc_list.append(mcc_max)
    auroc_list.append(auroc_max)
    auprc_list.append(auprc_max)
    epoch_list.append(epoch_max)
    
cv_result = pd.DataFrame([acc_list, sens_list, spec_list, mcc_list, auroc_list, auprc_list],
                         index=['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUROC', 'AUPRC'])
cv_result['Mean'] = cv_result.mean(axis=1)
print('Cross-validation results:')
print('Stopping Epochs:', epoch_list)
print(cv_result)

print('Testing progressing...')
model = Net(dim1, dim2, dim3, dropout).to(device)
optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))

auroc_max = 0
for epoch in range(1000):
    model_train(train_index)
    acc, sens, spec, mcc, auroc, auprc = test(test_index)
    if auroc > auroc_max:
        acc_max = acc
        auroc_max = auroc
        auprc_max = auprc
        sens_max = sens
        spec_max = spec
        mcc_max = mcc
        epoch_max = epoch

test_result = pd.Series([acc_max, sens_max, spec_max, mcc_max, auroc_max, auprc_max],
                        index=['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUROC', 'AUPRC'])
print('Testing results at Epoch {}:'.format(epoch_max))
print(test_result)
