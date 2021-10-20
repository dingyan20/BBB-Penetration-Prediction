import pandas as pd

df_drug = pd.read_csv('drug_list_all.csv')
df_drug['Node_Idx'] = range(df_drug.shape[0])
dict_drug = df_drug.set_index('CID')['Node_Idx'].to_dict()

# drug-protein interactions
df_stitch = pd.read_csv(
    '9606.protein_chemical.links.detailed.v5.0.tsv', sep='\t')
df_protein = pd.read_csv('9606.protein.info.v11.5.txt', sep='\t')
df_stitch['chemical'] = df_stitch['chemical'].str[4:].astype(int)
df_protein = df_protein.rename(columns={'#string_protein_id': 'protein'})

df_drugpro = df_stitch[df_stitch['chemical'].isin(df_drug['CID'])]
df_drugpro = df_drugpro[df_drugpro['combined_score']
                        >= 500].reset_index(drop=True)
df_drugpro = df_drugpro.join(df_protein.set_index('protein'), on='protein')
df_drugpro = df_drugpro.dropna().reset_index(drop=True)
df_drugpro.rename(columns={'chemical': 'Drug_ID',
                           'preferred_name': 'Gene_ID'}, inplace=True)

# types of drug-protein interactions
df_drugpro['Relation'] = [4]*df_drugpro.shape[0]
df_drugpro['Relation'][df_drugpro['Gene_ID'].str.contains('ABC')] = 2
df_drugpro['Relation'][df_drugpro['Gene_ID'].str.contains('SLC')] = 1

# carrier proteins
df_drkg = pd.read_csv('drkg.tsv', sep='\t', header=None)
df_drkg = df_drkg[df_drkg[1] == 'DRUGBANK::carrier::Compound:Gene']
carrier_gene_id = df_drkg[2].str[6:].unique()

df_hugo = pd.read_csv('HUGO.txt',sep='\t', usecols=[1, 2, 4, 5, 11])
df_hugo = df_hugo.dropna(subset=['NCBI Gene ID'])
df_hugo['NCBI Gene ID'] = df_hugo['NCBI Gene ID'].astype(int)
carrier_gene = df_hugo[df_hugo['NCBI Gene ID'].astype(
    str).isin(carrier_gene_id)]['Approved symbol']
carrier_gene = carrier_gene[~(carrier_gene.str.startswith(
    'SLC') | carrier_gene.str.startswith('ABC'))].values

df_drugpro['Relation'][df_drugpro['Gene_ID'].isin(carrier_gene)] = 3

df_drugpro = df_drugpro[~df_drugpro.duplicated(
    subset=['Drug_ID', 'Gene_ID'])].reset_index(drop=True)

gene_node = df_drugpro['Gene_ID'].unique()
df_gene_node = pd.DataFrame({'Gene_ID': gene_node, 'Node_Idx': range(
    df_drug.shape[0], df_drug.shape[0]+len(gene_node))})
dict_gene = df_gene_node.set_index('Gene_ID')['Node_Idx'].to_dict()
df_drugpro['Drug_ID'] = df_drugpro['Drug_ID'].map(dict_drug)
df_drugpro['Gene_ID'] = df_drugpro['Gene_ID'].map(dict_gene)

df_drugpro.to_csv('drug_protein_interaction.csv')
