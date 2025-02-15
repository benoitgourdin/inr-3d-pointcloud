# Implicit Neural Representations of 3D Medical Point Clouds

## Experiment Scripts and Datasets

This repository contains various scripts for conducting experiments related to shape reconstruction and registration. Below is an overview of the scripts used for different types of experiments and the datasets utilized.

### Experiment Scripts

- **SIREN-based Mesh Data Tests**
  - Script: `siren/experiment_scripts/train_sdf.py`
  - Description: This script is used for testing on mesh data with **SIREN**.

- **Sparse Shape Reconstruction on Mesh Data**
  - Script: `inr-implicit-shape-reconstruction-mesh/src/impl_recon/train.py`
  - Description: This script is used for conducting tests on **sparse shape reconstruction** with mesh data.

- **Pairwise Registration Experiments on Point Clouds**
  - Script: `registration-pipeline/src/train.py`
  - Description: This script is utilized for **pairwise registration** experiments.

- **Cohort-Based Registration Experiments on Point Clouds**
  - Script: `registration-pipeline-latent/src/train.py`
  - Description: This script is used for **cohort-based registration** experiments.

### Datasets Used

- **MedShapeNet Dataset**
  - Used for **SIREN** and **Sparse Shape Reconstruction** experiments.
  - https://medshapenet-ikim.streamlit.app

- **DeformingThings4D**
  - Used for **Registration Experiments**.
  - https://github.com/rabbityl/DeformingThings4D

- **Lung250M-4B**
  - Used for **Registration Experiments**.
  - https://cloud.imi.uni-luebeck.de/s/s64fqbPpXNexBPP

For further details on running the experiments, refer to the respective script documentation or comments within the code files.
