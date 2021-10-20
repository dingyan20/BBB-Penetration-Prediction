import pandas as pd
import numpy as np
import torch
import deepchem as dc
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

df_drug = pd.read_csv('drug_list_all.csv')

# edges
df_drugsim = pd.read_csv('drug_similarity.csv')
df_drugpro = pd.read_csv('drug_protein_interaction.csv')

sourcenode = df_drugsim['Drug_1'].to_list() + df_drugpro['Drug_ID'].to_list()
destinnode = df_drugsim['Drug_2'].to_list() + df_drugpro['Gene_ID'].to_list()
edge_index = torch.tensor(
    [sourcenode + destinnode, destinnode + sourcenode], dtype=torch.long)
relation_list = df_drugsim['Relation'].to_list() + \
    df_drugpro['Relation'].to_list()
edge_type = torch.tensor(relation_list*2, dtype=torch.long)

# node features
# Mordred descriptors
featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
mordred_des = featurizer.featurize(df_drug['CanonicalSMILES'].to_list())
mordred_des = StandardScaler().fit_transform(mordred_des)
mordred_des = torch.tensor(mordred_des, dtype=torch.float)

# DRKG embeddings
df_drug['Entity'] = 'Compound::' + df_drug['DRKG_ID']
df_drug_drkg = df_drug[df_drug['DRKG_ID'].notna()]
df_drug_drkg.iloc[1, -1] = 'Compound::chebi:134923'

drkg_emb = np.load('DRKG_TransE_l2_entity.npy')
df_emb_map = pd.read_csv('entities.tsv',
                         sep='\t', names=['Entity', 'Emb_Idx'])

df_drug_drkg = df_drug_drkg.join(df_emb_map.set_index('Entity'), on='Entity')
drkg_emb = drkg_emb[df_drug_drkg['Emb_Idx']]
drkg_emb = StandardScaler().fit_transform(drkg_emb)
drkg_emb = torch.tensor(drkg_emb, dtype=torch.float)

y = torch.tensor(df_drug['P_NP'].values)

# graph with both drug-protein interactions and drug-drug similarity as the edges
# Mordred descriptors as the node features
data_drugsim = Data(x=mordred_des, edge_index=edge_index,
                    edge_type=edge_type, y=y)
torch.save(data_drugsim, 'graph_drugsim.pt')

# graph with drug-protein interactions and drug-drug similarity as the edges
# Mordred descriptors and DRKG embeddings as the node features
data_drugsim_drkg = Data(x=mordred_des, x1=drkg_emb, edge_index=edge_index,
                         edge_type=edge_type, y=y)
torch.save(data_drugsim_drkg, 'graph_drugsim_drkg.pt')

# graph with only drug-protein interactions as the edges
# Mordred descriptors as the node features
edge_type_sub = edge_type[(edge_type != 0)]
edge_index_sub = edge_index[:, (edge_type != 0)]
edge_type_sub[edge_type_sub == 4] = 0
data = Data(x=mordred_des, edge_index=edge_index_sub, edge_type=edge_type_sub, y=y)
torch.save(data, 'graph.pt')
