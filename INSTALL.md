instruction that worked with my setup 

ubuntu 22.04 LTS, RTX 3050-mobile GPU, CUDA 11.3
GCC (g++) 9.5.0, 

For details, please check official pytorch3d repo https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
If you need to ensure that conda base env is clean and consistent 

```
conda activate base
conda clean --all
conda update --all
```

Ensure this does not throw any Conda verification or cobbler errors. No with clean-slate conda, you can install pytorch3d as follows:

1. Create a conda environment and install pytorch with cuda support.
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Check if torch (py3.9_cuda11.6_cudnn8.3.2_0) is installed correctly and "IS_CUDA_AVAILABLE flag is true.

```
python collect_env.py 
```

2. Install util packages-
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

```

3. Since collect_env.py showed that I have  CUDA 11.6.124 runtime, I followed the step to install  the CUB build time dependency, which you only need if you have CUDA older than 11.7, 
```
conda install -c bottler nvidiacub
```

4. Follwing step throws several CondaVerificationErrors for packages in my setup:
`conda install jupyter # EROORS!`

So I installed jupyter with pip3

```
pip3 install --upgrade pip
pip3 install jupyter
```

5. Install necessary packages to run the code examples and test cases.
```
pip install scikit-image matplotlib imageio plotly opencv-python
```

6. Install pytorch3d and test installation
```
conda install pytorch3d -c pytorch3d
python
>>>import pytorch3d
>>>
```


