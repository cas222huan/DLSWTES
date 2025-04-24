# DL-SW-TES
Official codes and data for the paper entitled "Deep learning coupled with split window and temperature-emissivity separation (DL-SW-TES) method improves clear-sky high-resolution land surface temperature estimation".

If our work is helpful to you, please kindly cite our paper as:
```
@article{ZHANG2025,
title = {Deep learning coupled with split window and temperature-emissivity separation (DL-SW-TES) method improves clear-sky high-resolution land surface temperature estimation},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {225},
pages = {1-18},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.04.016},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001534},
author = {Huanyu Zhang and Tian Hu and Bo-Hui Tang and Kanishka Mallick and Xiaopo Zheng and Mengmeng Wang and Albert Olioso and Vincent Rivalland and Darren Ghent and Agnieszka Soszynska and Zoltan Szantoi and Lluís Pérez-Planells and Frank M. Göttsche and Dražen Skoković and José A. Sobrino}
}
```
Feel free to visit [my personal website](https://cas222huan.github.io//) for more information regarding LST retrieval and thermal infrared remote sensing.
## Filetree
```
.
├── data/: atmospheric profiles and parameters, spectra library, simulation dataset, site measurement, etc
├── lib/: self-defined functions
├── log/: log of each epoch during model training
├── matlab_plot/: matlab codes and data for density scatter
├── model:/ trained ResMLP and Catboost models
├── result:/ radiance and LST estimates on the simulation dataset
├── dataset_build.ipynb: construct the simulation dataset for model training and evaluation
├── run_rttov.ipynb: run the RTTOV model for atmospheric correction
├── model_train.ipynb: train the ResMLP and Catboost(+Optuna) models
├── simulation_analysis.ipynb: evaluate the performances of DL-SW-TES, TES, and Catboost on the simulated test set
├── sw_tes_like.ipynb: reproduce the SW-TES and SWDTES algorithms and evaluated their performances on the simulated test set
├── T-based_validation.ipynb: evaluate the accuracy of TES, DL-SW-TES, SW-TES, and SWDTES using in-situ measurements
├── cross_validation.ipynb: evaluate the performances of DL-SW-TES (as well as SW-TES and SWDTES) by intercomparing with NASA's official ECO2LSTE product
├── requirements.txt: python packages used in this project (scanned by pipreqsnb)
└── README.md
```

## Note
* Certain sections of the code (e.g., the TES algorithm, interpolation of ERA5 atmospheric profiles to match ECOSTRESS pixels) are not included. These codes are copyrighted by the European ECOSTRESS Hub (EEH) and we do not have the permission to share them.
* Due to differences in hardware (e.g., GPU, CUDA versions) and software environments (e.g., Python packages), exact replication of the PyTorch optimization results may not be guaranteed across different devices. However, the final performance is expected to remain similar.
* Data and results of this project are provided in https://drive.google.com/file/d/1IBScq_MzSLVwkHtxoRs-dL6-bMHKR8Cf/view?usp=sharing. In-situ LST measurements from the KIT & Copernicus network (PI: Prof. Frank M. Göttsche) and the GCU network (PI: José A. Sobrino) are not publicly available. Please contact the respective PIs for access.