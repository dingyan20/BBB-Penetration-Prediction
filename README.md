# RGCN for predicting BBB Penetration of drug molecules

This repository contains the data and the codes for the manuscript "Relational graph convolutional networks for predicting blood-brain barrier penetration of drug molecules".

### Getting started

1. Run **get_large_files.sh** to download the big data files.

2. Calculate the drug-drug similarity by running **drug_similarity.py**. The results will be stored in **drug_similarity.csv**.

3. Collect the drug-protein interactions with **drug_protein_interaction.py**. The results will be saved in **drug_protein_interaction.csv**.

4. Run **graph&#46;py** to generate the drug features and to structure the data into graphs. Three graphs will be built and be saved separately in the following files.

    - **graph&#46;pt** includes the drug-protein interactions as the edges and the Mordred descriptors as the node features.

    - **graph_drugsim.pt** includes the drug-protein interactions and the drug-drug similarity as the edges, and the Mordred descriptors as the node features.

    - **graph_drugsim_drkg.pt** includes the drug-protein interactions and the drug-drug similarity as the edges. The Mordred descriptors and the DRKG embeddings were combined and used as the node features.

5. Run **rgcn&#46;py** to train and evaluate the RGCN model with **graph&#46;pt** as the input. Similarly, **rgcn_drugsim.py** and **rgcn_drugsim_drkg.py** train and evaluate the RGCN model using **graph_drugsim.pt** and **graph_drugsim_drkg.pt** as the input, respectively.

