## 2. 프로젝트 구조
```sh
.
├── model
│   ├── fine_tune_gnn.py
│   ├── fine_tune_sts.py
│   └── SimCSE.py
├── preprocessing
│   ├── modeling
│   │   └── Clustering.ipynb
│   ├── DataCleaning.ipynb
│   ├── EDA.ipynb
│   ├── v1_downsampling.ipynb
│   ├── v2_augmentation_biassed.ipynb
│   ├── v3_augmentation_uniform.ipynb
│   └── v4_augmentation_spellcheck.ipynb
├── resources
│   ├── log
│   └── raw
│       ├── dev.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── utils
│   ├── data_module.py
│   ├── ensemble_module.py
│   └── helpers.py
├── inference.py
├── run_ensemble.py
├── train_graph.py
├── train.py
├── train_unsup_CL.py
```
