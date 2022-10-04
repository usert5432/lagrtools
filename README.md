# lagrtools -- Tools to handle Wire-Cell Graphs

`lagrtools` provides users with tools to handle Wire-Cell graphs and convert
them to a format acceptable for GNN libraries (e.g. `torch_geometric`).

# Installation
`toytools` is intended for developers, so the best way to install it is to run
```
python setup.py develop
```

# Overview

As of now, `lagrtools` has several components:

1. A converter script `srcipts/preprocess` that takes a Wire-Cell graph dataset
   (made of pairs of graphs `(img, tru)`, c.f. `Data Format`) and transforms
   it into a simple dataset of heterogeneous graphs suitable for
   `torch_geometric`.

2. A helper script `srcipts/dataset_stats` that calculates various statistics
   of node features.

3. `torch_geometric` dataset in `lagrtools/torch`.

4. `torch_geometric` transformations in `lagrtools/torch/transforms`.


## Wire-Cell Graph Format

Currently, Wire-Cell outputs graphs as pairs of `(img, tru)`. Each graph in a
pair conforms to the following
[schema](https://github.com/WireCell/wire-cell-toolkit/blob/master/aux/docs/ClusterArrays.org).

The `img` graph contains reconstructed information. This info is supposed to be
fed as input to Graph Neural Networks (GNN). The `tru` graph contains truth
information that the GNN is expected to predict.

`tru` and `img` graph may have different numbers of nodes or edges. `lagrtools`
will try to match these graphs according to the node `ident` field. The nodes
that are present in one graph and absent in another will be pruned.

`lagrtools` support selection a subset of all features/nodes from the graphs
according to a configuration file. An example of such a configuration file can
be found here `examples/preprocess_configs/simple.toml`.


## Usages Examples

### 1. Converting Wire-Cell Graph Dataset to a torch_geometric Format

The script `scripts/preprocess` can be used to convert the raw Wire-Cell
dataset into a simpler format that is suitable for `torch_geometric`.
This script can be run as:

```
$ python3 scripts/preprocess $INPUT $OUTPUT --config $CONFIG
```

Here, `$INPUT` is a path to a directory where the raw Wire-Cell dataset is
located. The dataset is a collection `.npz` files with names
`clusters-(tru|img)-aN.npz`.

`$OUTOUT` is an output directory where the processed dataset will
be saved. Finally, `$CONFIG` is a path to a config, that defines which
nodes/features from the Wire-Cell graphs to extract. Please, refer to the
example file `examples/preprocess_configs/simple.toml` for details.


### 2. Using Converted Dataset

A dataset, created by `scripts/preprocess` can be loaded to a `HeteroData`
structure with a help of a `LAGRDataset` dataset. Its constructor expects
a path to the converted dataset, and, optionally, a list of additional graph
transformations to apply.

