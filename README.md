# RGCN for predicting BBB Penetration of drug molecules

- **get_large_files.sh** downloads the large data files.

- **drug_similarity.py** calculates the drug-drug similarity, which is stored in **drug_similarity.csv**.

- **drug_protein_interaction.py** collects the drug-protein interactions and save them in **drug_protein_interaction.csv**.

- **graph&#46;py** generates the drug features and structures the data into three graphs, which are stored in the following files.

  - **graph&#46;pt** includes the drug-protein interactions as the edges and the Mordred descriptors as the node features.

  - **graph_drugsim.pt** includes the drug-protein interactions and the drug-drug similarity as the edges, and the Mordred descriptors as the node features.

  - **graph_drugsim_drkg.pt** includes the drug-protein interactions and the drug-drug similarity as the edges. The Mordred descriptors and the DRKG embeddings were combined and used as the node features.

- **rgcn&#46;py** trains and evaluates the RGCN model using **graph&#46;pt** as the input.

- **rgcn_drugsim.py** trains and evaluates the RGCN model using **graph_drugsim.pt** as the input.

- **rgcn_drugsim_drkg.py** trains and evaluates the RGCN model using **graph_drugsim_drkg.pt** as the input.
