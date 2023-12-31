# p-Poisson surface reconstruction in curl-free flow from point clouds

This repository contains the official Pytorch implementation for the **NeurIPS 2023 paper "p-Poisson surface reconstruction in curl-free flow from point clouds"**

by Yesom Park, Taekyung Lee, Jooyoung Hahn, and Myungjoo Kang

---

<p align="center">
  <img src="https://github.com/Yebbi/PINC/assets/82932461/2769a834-800c-4500-9168-74b4f31708a4" width=700 />
</p>

We propose a new model for the implicit reconstruction of 3D surfaces by learning signed distance functions (SDFs) only from raw point clouds, without the need for ground truth distances or point normals. The proposed model aims to show that adequate supervision of partial differential equations and fundamental characteristics of differential vector fields suffices for the effective reconstruction of high-quality surfaces. To facilitate precise learning of the SDF, a variable splitting structure is utilized by integrating a gradient of the SDF as an auxiliary variable. The $p$-Poisson equation is incorporated as a hard constraint in the auxiliary variable, with its unique weak solution converging to the SDF as $p$ approaches infinity.
Moreover, in order to improve the accuracy of the reconstruction, a constraint for curl-free conditions is applied to the auxiliary variable, utilizing the irrotational property of the conservative vector field.
All combined, we achieved superior and robust reconstruction only from point clouds without any priori knowledge of the surface normal at the data points.


## How to run the code

### Dependencies
This code is developed with Python3. PyTorch >=1.8 (we recommend 1.10.0). First, install the dependencies by running the following to install a subset of the required Python packages in place
```
pip install -r requirements.txt
```

### Usage

Train and evaluate our model through `reconstruction/run.py`

To **train** the model from scratch, simply run the following command

```
python reconstruction/run.py
  --config: Training configuration
  --expname: The folder name for storing evaluation results
    (default: 'single_shape')
```
* `config` is the path to the config file. The prescribed config files are provided as `setup.conf` in `./reconstruction` andt he default configurations are provided.

* When trained, the model is saved in a folder `exps` in your workdir. `expname` is the name of a subfolder in `exps` that stores the results.

To **evaluate** a trained model on an epoch EPOCH, use the following

```
python reconstruction/run.py --eval --checkpoint EPOCH
```



