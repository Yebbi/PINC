import os

import numpy as np
import torch
import trimesh
import random
import open3d as o3d

def save_configs(directory, is_continue, otherdir = False,**configs):
    if is_continue and not(otherdir) :
        return
    else : 
        f = open(f'{directory}/config.txt', 'a')
        for i in configs:
            f.write(f'{i}: {configs[i]} \n' )
        f.close()
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def load_point_cloud_by_file_extension(file_name):

    ext = file_name.split('.')[-1]

    point_set1 = torch.tensor(o3d.io.read_point_cloud(file_name, ext).points).float()
    center = torch.mean(point_set1,0)
    point_set1 = point_set1 - center
    max_pts = torch.abs(point_set1).max()
    point_set1 = point_set1 / max_pts
    
    point_set2 = torch.tensor(o3d.io.read_point_cloud(file_name, ext).normals).float()
    point_set = torch.concat([point_set1, point_set2], dim=-1)
       
    return point_set, max_pts, center


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)
    
def bumpft(val, epsilon=0.1):
    return 1-torch.tanh(val/epsilon)**2