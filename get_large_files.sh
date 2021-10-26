#! /bin/bash

echo 'Downloading drug-protein interactions from STITCH...'
wget http://stitch.embl.de/download/protein_chemical.links.detailed.v5.0/9606.protein_chemical.links.detailed.v5.0.tsv.gz
gzip -d 9606.protein_chemical.links.detailed.v5.0.tsv.gz
echo -e 'Done.\n\n'

echo 'Downloading data from DRKG...'
wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
tar -xzvf drkg.tar.gz
mv embed/DRKG_TransE_l2_entity.npy .
rm entity2src.tsv relation_glossary.tsv drkg.tar.gz
rm ._drkg.tsv ._embed ._entity2src.tsv ._relation_glossary.tsv
rm -r embed
echo -e 'Done.\n\n'