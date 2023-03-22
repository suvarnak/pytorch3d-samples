This repo is pytorch3D implementations for few common 3D Vision tasks

1. Deform a source point cloud  to form a target point cloud using 3D loss function - Chamfer Distance

You may use any obj file as target or download (see download.sh) a sample dolphin.obj

```
conda activate pytorch3d
python sphere_to_object.py -target input/dolphin.obj
```

sphere_to_object.py is extended from PyTorch3D official tutorial to perform following -

The script 'sphere_to_object' learns a mapping function that can take a some point cloud as input, for example, a sphere and generate a specific target object.

The function learns to incrementally deform a source mesh (sphere) to be more similar to the target mesh (e.g dolphin) and use  a 3D loss functions, such as 
chamfer distance to guide the learning.

Intermediate generation results are stored as a 2D image and finally a GIF of the generated point clouds is created to visualize the incremental transformation from a sphere point cloud to a desired object.
