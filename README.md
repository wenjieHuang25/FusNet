# FusNet
## A deep learning method based on model fusion strategy for predicting protein-mediated loops.
## Contents
- [Installation](#Installation)
- [Model training](#Model_training)
- Model predicting


## <span id="Installation">Installation</span>
FusNet is built on Python3. tombo is required to re-squiggle the raw signals from nanopore reads before running deepsignal.

- Prerequisites:

Python 3.x

tensorflow

- Dependencies:

h5py

numpy

pandas

xgboost

lightgbm

scikit-learn

**1. Create an environment**
```
# create a new enviroment
conda create -n fusnet python=3.9
# activate
conda activate fusnet
# deactivate
conda fusnet
```

**2. Install FusNet**
```
git clone https://github.com/bioinfomaticsCSU/FusNet.git
# Using FusNet as the root directory for program execution
cd FusNet
```

## <span id="Model_training">Model training</span>
**1. Data preprocessing**
```
bash preprocess/preprocess_data.sh \
data/gm12878_rad21/gm12878_rad21_interactions_hg19.bedpe \
data/gm12878_rad21/gm12878_DNase_hg19.narrowPeak \
data/gm12878_rad21/gm12878_rad21_TF_hg19.narrowPeak gm12878_rad21 out_dir
```
After the program runs successfully, the following files will be generated
```

```

**2. Training sequence feature extractor**
```
python fusnet/train_feature_extractor.py out_dir/gm12878_rad21_loops \
gm12878_rad21_extractor out_dir
```
After the program runs successfully, the following files will be generated
```

```

**3. Feature extracting**
```
python fusnet/extract_feature.py \
                  out_dir/gm12878_rad21_extractor.model.pt \
                  out_dir/gm12878_rad21_loops \
                  gm12878_rad21_extracted \
                  out_dir;
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_extracted_train_factors_outputs.hdf5   
out_dir/gm12878_rad21_extracted_valid_factor_outputs.hdf5
out_dir/gm12878_rad21_extracted_test_factor_outputs.hdf5
```

**4. Training fusion model classifiers**
```
python fusnet/train_fusion_model.py out_dir gm12878_rad21
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_knn_predictor.pkl 
out_dir/gm12878_rad21_lgb_predictor.pkl 
out_dir/gm12878_rad21_xgb_predictor.pkl
out_dir/gm12878_rad21_rf_predictor.pkl 
```

**5. Loops predicting**
```
python fusnet/predict.py -m out_dir/gm12878_rad21_extractor.model.pt \
                --data_name gm12878_rad21 \
                --data_file out_dir/gm12878_rad21_extracted_test_factor_outputs.hdf5 \
                --output_pre out_dir/gm12878_rad21_extracted_test
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_extracted_test_FusNet_probs.txt
```

## Model predicting{#Model predicting}
**1. Data preprocessing**
```
python preprocess/generate_loops.py -m 1000 -e 500 \
--positive_loops out_dir/gm12878_rad21_negative_loops.bedpe \
-r data/hg19.fa -n gm12878_rad21_loops_negative -o out_dir -p True
```

**2. Extracting sequence features**
```
python fusnet/extract_feature.py \
  out_dir/gm12878_rad21_extractor.model.pt \
  out_dir/gm12878_rad21_loops_all_predict.hdf5 \
  gm12878_rad21_extracted_all_predict_ \
  out_dir;
```

**3. Loops predicting**
```
python fusnet/extract_feature.py \
  out_dir/gm12878_rad21_extractor.model.pt \
  out_dir/gm12878_rad21_loops_all_predict.hdf5 \
  gm12878_rad21_extracted_all_predict_ \
  out_dir;
```
