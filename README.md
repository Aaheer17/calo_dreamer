# CaloDream

A repository for fast detector simulation using Conditional Flow Matching.
This is the reference code for CaloDREAM [arXiv:2405.09629](https://arxiv.org/abs/2405.09629) and the corresponding 
entry in the Calochallenge.

Samples used for the evaluation of the networks are published in the Zenodo repository 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14413047.svg)](https://doi.org/10.5281/zenodo.14413047)

## Usage

Training:
```
python3 src/main.py path/to/yaml --use_cuda
```

The documenter will create a folder in `results` with the date as
prefix and the specified `run_name`.

### Parameters

Parameter		| Usage
------------------------| ----------------------------------------
run\_name		| Name of the output folder
hdf5\_file		| Path to the .hdf5 file used for training
xml\_filename		| Path to the .xml file used to extract the binning information
p\_type 		| "photon", "pion", or "electron"
dtype			| specify default dtype
eval\_dataset		| "1-photons", "1-pions", "2", or "3" used in the CaloChallenge evaluation
model\_type         | Model to be trained: "energy" or "shape" 
network             | Type of network (see Networks for more details)

### Training parameters

Parameter 		| Usage
------------------------| ----------------------------------------
dim			| Dimensionality of the input
n\_con			| Number of conditions
width\_noise		| Noise width used for the noise injection
val\_frac		| Fraction of events used for validation
transforms		| Pre-processing steps defined as an ordered dictionary (see transforms.py for more details)
lr			| learning rate
max\_lr			| Maximum learning rate for OneCycleLR scheduling
batch\_size		| batch size
validate\_every		| Interval between validations in epochs
use\_scheduler 		| True or False
lr\_scheduler		| string that defines the learning rate scheduler
cycle\_epochs		| defines the length of the cycle for the OneCycleLR, default to # of epochs
save\_interval		| Interval between each model saving in epochs
n\_epochs		| Number of epochs

### ResNet parameters

Parameter		| Usage
------------------------|----------------------------------------
intermediate\_dim	| Dimension of the intermediate layer
layers\_per\_block	| Number of layers per block
n\_blocks		| Number of blocks
conditional		| True/False, it should be always True

An example yaml file is provided in `./configs/cfm_base.yaml`.

### ViT parameters

Parameter       | Usage
------------------------|----------------------------------------
patch\_shape        | Shape of a single patch
hidden\_dim         | Hidden/Embedding dimension
depth               | Number of ViT blocks
num\_heads          | Number of transformer heads
mlp\_ratio          | Multiplicative factor for the MLP hidden layers
learn\_pos\_embed   | (True or False) learnable position embedding

An example yaml file is provided in `./configs/d2_shape_model_vit.yaml`

Plotting:

To run the sampling and the evaluation of a trained model.
```
python3 src/main.py --use_cuda --plot --model_dir path/to/model --epoch model_name
```


## Energy-Stratified Configurations (Dataset 2)

The following configuration files correspond to stratified incident energy splits for Dataset 2.

| Energy Bin | Train Data | Test Data | Config File |
|------------|------------|-----------|-------------|
| **High**   | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_train_high.hdf5` | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_test_high.hdf5` | `configs/d2_shape_Baseline_high.yaml` |
| **Mid**    | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_train_mid.hdf5`  | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_test_mid.hdf5`  | `configs/d2_shape_Baseline_mid.yaml`  |
| **Low**    | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_train_low.hdf5`  | `/project/biocomplexity/Calorimeter/stratified_by_energy_groups_no_split/dataset_2_test_low.hdf5`  | `configs/d2_shape_Baseline_low.yaml`  |

Each configuration file is tuned to the corresponding incident energy range and should be used only with its matching stratified dataset.

**Important:**  
For each configuration file, update **lines 6, 11, and 23** to reflect the correct paths to:
- the XML file  
- the energy model  
- the XML file

These paths are user-dependent and must be adapted before execution.
