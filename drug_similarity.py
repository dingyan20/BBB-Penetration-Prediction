import os
import pandas as pd
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit import Chem
from rdkit.Chem.Pharm2D import Generate
from rdkit import DataStructs

df_drug = pd.read_csv('drug_list_all.csv')

# pharmacophore fingerprint
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
sigFactory = SigFactory(factory, minPointCount=2,
                        maxPointCount=3, skipFeats=['ZnBinder'])
sigFactory.SetBins([(0, 2), (2, 4), (4, 6), (6, 10)])
sigFactory.Init()

mols = [Chem.MolFromSmiles(x) for x in df_drug['CanonicalSMILES'].to_list()]
fps = [Generate.Gen2DFingerprint(mol, sigFactory) for mol in mols]

# drug-drug similarity
sim_scores = []
for i in range(len(fps)-1):
    for j in range(i+1, len(fps)):
        score = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sim_scores.append([i, j, score])

df_sim = pd.DataFrame(sim_scores, columns=['Drug_1', 'Drug_2', 'Similarity'])
df_drugsim = df_sim[df_sim['Similarity'] > 0.7].reset_index(drop=True)
df_drugsim['Relation'] = [0]*df_drugsim.shape[0]

df_drugsim.to_csv('drug_similarity.csv', index=False)
