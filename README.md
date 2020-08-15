# BEULAR-NB

Model code for the SageMaker instance of the [beular-api](https://github.com/csmcallister/beular-api) project.

## Getting Started

Ensure that this repo is public and add it's git url to `config.py` in beular-api. This will automatically git clone this project into the SageMaker instance once it builds. It will also let you source any code you develop while working within SageMaker.

### Downloading the BlazingText Model

The BlazingText model is around 1.2GB, so it's not suitable for GitHub. You can download that model from [here](https://drive.google.com/file/d/1G-oWjBq5iuD6N8b7fiaSo-hVhSZ_Z1WE/view?usp=sharing). Place it in the `pipeline/estimators/` directory with the other models. Once you're up and running with AWS, this (and other models) will be stored in S3.
