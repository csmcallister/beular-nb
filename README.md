# BEULAR-NB

Model code for the SageMaker instance of the [beular-api](https://github.com/csmcallister/beular-api) project.

## Getting Started

These notebooks, with the exception of `pipeline/Analysis.ipynb` and `pipeline/Eval.ipynb` are meant to be run in an AWS Sagemaker Instance. If working with the former two noteboks, install the requirements with `pip install -r requirements.txt`.

If you're connecting this repo to the SageMaker instance in [beular-api](https://github.com/csmcallister/beular-api), ensure that this repo is public and add it's git url to `config.py` in beular-api. This will automatically git clone this project into the SageMaker instance once it builds. It will also let you source any code you develop while working within SageMaker.

### Downloading the BlazingText Model

The BlazingText model is around 1GB, so it's not suitable for GitHub. You can download that model from [here](https://drive.google.com/file/d/16EG0Zfj-ChdzM_R_W9cBKEHxpcNYMSku/view?usp=sharing). Place it in the `pipeline/estimators/` directory with the other models. Once you're up and running with AWS, this (and other models) will be stored in S3.
