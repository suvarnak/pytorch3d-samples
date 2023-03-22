import numpy as np
import datetime
from pytorch3d.io import load_obj, save_obj
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
import argparse
import torch
need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


class sphere_to_object(object):
    """ 
    Code to learn mapping function that can take a sphere point cloud as input and generate a specific object
    The train loop incrementally deform a source mesh (sphere) and compare it with a target mesh (e.g dolphin) using 3D loss functions. 


    """

    def __init__(self, args, device):
        self.device = device
        self.target_object_file_name = args.target
        self.generate_object_file_name = "generated_" + args.target
        self.save_generations = args.save
        self.initialize_net()
        self.plot_pointcloud(self.trg_mesh, "Target mesh")
        self.plot_pointcloud(self.src_mesh, "Source mesh")
        self.train()
        self.write_gif()

    def __str__(self):
        return 'name@{:#x}: {}'.format(id(self), self.target_object_file_name)

    def initialize_net(self):
        trg_obj = self.target_object_file_name  # load the target mesh
        # read the target 3D object  using load_obj
        verts, faces, aux = load_obj(trg_obj)
        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)
        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        self.center = verts.mean(0)
        verts = verts - self.center
        self.scale = max(verts.abs().max(0)[0])
        verts = verts / self.scale
        # We construct a Meshes structure for the target mesh
        self.trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
        # We initialize the source shape to be a sphere of radius 1
        self.src_mesh = ico_sphere(4, self.device)
        # We will learn to deform the source mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices in src_mesh

    def plot_pointcloud(self, mesh, title="Generated PCD"):
        # Sample points uniformly from the surface of the mesh.
        points = sample_points_from_meshes(mesh, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        plt.savefig('./output/'+title +'.png')
        plt.close()

    def plot_losses(self, chamfer_losses, edge_losses, normal_losses, laplacian_losses):
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(chamfer_losses, label="chamfer loss")
        ax.plot(edge_losses, label="edge loss")
        ax.plot(normal_losses, label="normal loss")
        ax.plot(laplacian_losses, label="laplacian loss")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16")

    def train(self):
        deform_verts = torch.full(self.src_mesh.verts_packed(
        ).shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
        # Number of optimization steps
        iter = 2000 
        # Weight for the chamfer loss
        w_chamfer = 1.0
        # Weight for mesh edge loss
        w_edge = 1.0
        # Weight for mesh normal consistency
        w_normal = 0.01
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.1
        # Plot period for the losses
        plot_period = 10

        loop = tqdm(range(iter))
        chamfer_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []

        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            # Deform the mesh
            new_src_mesh = self.src_mesh.offset_verts(deform_verts)
            # We sample 5k points from the surface of each mesh
            sample_trg = sample_points_from_meshes(self.trg_mesh, 5000)
            sample_src = sample_points_from_meshes(new_src_mesh, 5000)
            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)

            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(
                new_src_mesh, method="uniform")

            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + \
                loss_normal * w_normal + loss_laplacian * w_laplacian

            # Print the losses
            loop.set_description('total_loss = %.6f' % loss)

            # Save the losses for plotting
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            normal_losses.append(float(loss_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))

            # Plot mesh
            if i % plot_period == 0:
                self.plot_pointcloud(new_src_mesh, title="generated_pcd_%d" % i)
                final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
                # Scale normalize back to the original target size
                final_verts = final_verts * self.scale + self.center
                #save_obj(str(i)+self.generate_object_file_name, final_verts, final_faces)

            #print(loss)
            # Optimization step
            loss.backward()
            optimizer.step()

        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

        # Scale normalize back to the original target size
        final_verts = final_verts * self.scale + self.center

        # Store the predicted mesh using save_obj
        save_obj(self.generate_object_file_name, final_verts, final_faces)

    def write_gif(self):
        frames = []
        import glob, os
        os.chdir("./output")
        for file in sorted(glob.glob("*.png"), key=os.path.getmtime):

            image = imageio.v2.imread(f'{file}')
            frames.append(image)
        imageio.mimsave('generated.gif', # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second

          

def main():
    parser = argparse.ArgumentParser(
        description="generate target object from an input sphere")
    parser.add_argument("-target", type=str,
                        help="name of taget object file", default="target.obj")
    parser.add_argument(
        "-save", type=bool, help="flag to set if you wish to save intermediate generations")
    args = parser.parse_args()
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    generator = sphere_to_object(args, device=device)


if __name__ == '__main__':
    main()
