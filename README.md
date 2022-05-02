# PLAYGROUND NETWORK AND ITS HOMOLOGY

This code is based on the [dnn-topology](https://github.com/cipriancorneanu/dnn-topology) 
repository of Cirpian Corneanu (MIT License as explained there in issues).

Among the additions and changes are:

-Introduction of playground neural network family and spherical synthetic data set random_loader.

-Analysis of output via MDS (plot.py) and GUDHI (bottleneck.py and pointcloud.py). DIPHA is no 
longer used, but most of the code related to it is intact.

## Setup

DIPHA is disabled in current version, so to install the requirements it suffices to
```
pip install -r requirements.txt
```

An example of a script that works for the Playground network:
```
python main.py --net playground --playground_layers 4 128 128 128 2 --dataset random --random_data_row_count 1024 --trial 0 --lr 0.0005  --n_epochs_train 100 --epochs_test "100" --graph_type functional --train 1 --build_graph 1
```

To use plot.py and gudhi-files, one needs to specify the epoch to study (files appear in plot folder)

(depending on your shell one might need to add '' around the layer dimensions)

Tested with pyenv environment version 3.7.9.

 