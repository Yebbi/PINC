# PINC
This repository contains the official Pytorch implementation for the **NeurIPS 2023 paper "p-Poisson surface reconstruction in irrotational flow from point clouds"**

by Yesom Park, Taekyung Lee, Jooyoung Hahn, and Myungjoo Kang

<p align="center">
  <img src="https://github.com/Yebbi/PINC/assets/82932461/b4ccb89c-2112-4c5b-b0e5-5b416ddb937d" />
</p>

We propose a new model for the implicit reconstruction of 3D surfaces by learning signed distance functions (SDFs) only from raw point clouds, without the need for ground truth distances or point normals. The proposed model aims to show that adequate supervision of partial differential equations and fundamental characteristics of differential vector fields suffices for the effective reconstruction of high-quality surfaces. To facilitate precise learning of the SDF, a variable splitting structure is utilized by integrating a gradient of the SDF as an auxiliary variable. The $p$-Poisson equation is incorporated as a hard constraint in the auxiliary variable, with its unique weak solution converging to the SDF as $p$ approaches infinity.
Moreover, in order to improve the accuracy of the reconstruction, a constraint for curl-free conditions is applied to the auxiliary variable, utilizing the irrotational property of the conservative vector field.
