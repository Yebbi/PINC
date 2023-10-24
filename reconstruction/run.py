import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_surface_eval
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d

class ReconstructionRunner:

    def run(self):
        
        utils.set_random_seed(1234) 
        
        print("running")
     
        self.data = self.data.cuda()
        self.data.requires_grad_()
        
        if self.eval:

            print("evaluating")
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.checkpoint))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes_eval(epoch=self.startepoch, path=my_path)
            return

        print("training")
        write = SummaryWriter(self.cur_log_dir)
        for epoch in range(self.startepoch, self.nepochs + 1):
            
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            
            cur_data = self.data[indices]
            
            mnfld_pnts = cur_data[:, :self.d_in]
            
            mnfld_sigma = self.local_sigma[indices]
            
            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch)

            # Change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)
        
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()

            ## Forward pass
            mnfld_outputs = self.network(mnfld_pnts, return_grad=True, return_auggrad=self.auggrad)
            nonmnfld_outputs = self.network(nonmnfld_pnts, return_grad=True, return_auggrad=self.auggrad)

            mnfld_pred = mnfld_outputs["SDF_pred"]
            mnfld_G = mnfld_outputs["grad_pred"]
            mnfld_G_tilde = mnfld_outputs['auggrad_pred']
            nonmnfld_pred = nonmnfld_outputs["SDF_pred"]
            nonmnfld_G = nonmnfld_outputs["grad_pred"]
            nonmnfld_G_tilde = nonmnfld_outputs['auggrad_pred']

            ## Compute grad
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

            ## Manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()

            ## Gradient Matching loss
            grad_loss = (((nonmnfld_grad - nonmnfld_G).norm(2, dim=-1)** 2).mean()*nonmnfld_pnts.shape[0] + ((mnfld_grad - mnfld_G).norm(2, dim=-1)** 2).mean()*mnfld_pnts.shape[0])/(nonmnfld_pnts.shape[0]+mnfld_pnts.shape[0])
            
            ## Minimal Area loss
            area_loss = ((utils.bumpft(nonmnfld_pred,epsilon=self.epsilon)*((nonmnfld_grad).norm(2, dim=-1))).mean()*nonmnfld_pnts.shape[0]+ (utils.bumpft(mnfld_pred, epsilon=self.epsilon)*((mnfld_grad).norm(2, dim=-1))).mean()*mnfld_pnts.shape[0])/(nonmnfld_pnts.shape[0]+mnfld_pnts.shape[0])
                      
            if self.regularizer_type == 'curl':
                # Curl loss
                H1 = gradient(mnfld_pnts, mnfld_G_tilde[:,0])
                H2 = gradient(mnfld_pnts, mnfld_G_tilde[:,1])
                H3 = gradient(mnfld_pnts, mnfld_G_tilde[:,2])
                curlG_x = H3[:,1] - H2[:,2]
                curlG_y = H1[:,2] - H3[:,0]
                curlG_z = H2[:,0] - H1[:,1]
                curl_loss_mnfld = (curlG_x**2 + curlG_y**2 + curlG_z**2).mean()
                del H1, H2, H3, curlG_x, curlG_y, curlG_z
                
                H1 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:,0])
                H2 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:,1])
                H3 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:,2])
                curlG_x = H3[:,1] - H2[:,2]
                curlG_y = H1[:,2] - H3[:,0]
                curlG_z = H2[:,0] - H1[:,1]
                curl_loss_nonmnfld = (curlG_x**2 + curlG_y**2 + curlG_z**2).mean()
                del H1, H2, H3, curlG_x, curlG_y, curlG_z
                
                curl_loss = (curl_loss_mnfld*mnfld_pnts.shape[0] + curl_loss_nonmnfld*nonmnfld_pnts.shape[0]) /(nonmnfld_pnts.shape[0]+mnfld_pnts.shape[0])
                
                # Matching two auxiliary variables
                G_matching = (((nonmnfld_G_tilde - nonmnfld_G).norm(2, dim=-1)** 2).mean()*nonmnfld_pnts.shape[0] + ((mnfld_G_tilde - mnfld_G).norm(2, dim=-1)** 2).mean()*mnfld_pnts.shape[0])/(nonmnfld_pnts.shape[0]+mnfld_pnts.shape[0])
            
            elif self.regularizer_type == 'none':
                curl_loss = torch.zeros(1)
                G_matching = torch.zeros(1)
                loss = mnfld_loss + self.regularizer_coord[0]*grad_loss + self.regularizer_coord[3]* area_loss

            
            loss = mnfld_loss + self.regularizer_coord[0]*grad_loss + self.regularizer_coord[1]*(G_matching) + self.regularizer_coord[2]*curl_loss + self.regularizer_coord[3]* area_loss
                
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    
            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)] Train Loss: {:.6f}\tManifold loss: {:.6f}'
                    '\tGrad loss: {:.6f}' '\tArea loss: {:.6f}' '\t {} loss: {:.6f}' '\tG_matching: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), grad_loss.item(), area_loss.item(), self.regularizer_type, tmp, G_matching.item()))
                
                write.add_scalar("Train loss", loss.item(), epoch)
                write.add_scalar("Manifold loss", mnfld_loss.item(), epoch)
                write.add_scalar("Grad loss", grad_loss.item(), epoch)
                write.add_scalar("Area_loss",  area_loss.item(), epoch)
                write.add_scalar("{} loss".format(self.regularizer_type), tmp, epoch)
                write.add_scalar("G_matching", G_matching.item(), epoch)
                write.add_scalar("learning rate", self.optimizer.param_groups[0]['lr'], epoch)
                f = open(f'{self.cur_exp_dir}/logs.txt', 'a')
                f.write('Train Epoch: [{}/{} ({:.0f}%)] Train Loss: {:.6f}\tManifold loss: {:.6f}'
                    '\tGrad loss: {:.6f}' '\tArea loss: {:.6f}' '\t {} loss: {:.6f}' '\tG_matching: {:.6f}\n'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), grad_loss.item(), area_loss.item(),self.regularizer_type, curl_loss.item(), G_matching.item()))
                f.close()
                    
            del cur_data, mnfld_pnts, nonmnfld_pnts, indices, mnfld_sigma, mnfld_pred, nonmnfld_pred, nonmnfld_G, nonmnfld_grad, mnfld_loss, grad_loss, loss, area_loss, mnfld_grad, mnfld_G, nonmnfld_G_tilde, G_matching, curl_loss, curl_loss_nonmnfld, curl_loss_mnfld
        
    def plot_shapes(self, epoch, path=None):
        # plot network validation shapes
        self.network.eval()
        #with torch.no_grad():
        
        if not path:
            path = self.plots_dir
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            pnts = self.data[indices, :3]
            plot_surface(with_points=True,
                            points=pnts,
                            decoder=self.network,
                            path=path,
                            epoch=epoch,
                            shapename=self.expname,
                            **self.conf.get_config('plot'))
        
        
            
    def plot_shapes_eval(self, epoch, path=None):
        # plot network validation shapes
        self.network.eval()
        #with torch.no_grad():
        
        indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
        pnts = self.data[indices, :3]
        if path:
            plot_surface_eval(with_points=True,
                            points=pnts,
                            decoder=self.network,
                            path=path,
                            shapename=self.expname,
                            scale = self.scale, 
                            center = self.center,
                            **self.conf.get_config('plot'))
        
    def __init__(self, **kwargs):
        self.home_dir = os.path.abspath(os.pardir)
        self.regularizer_type = args.regularizer_type.lower()
        self.auggrad= args.auggrad
        self.regularizer_coord = args.regularizer_coord
        self.epsilon = kwargs['epsilon']
        
        # config setting
        if type(kwargs['conf']) == str:
            self.conf_filename = './reconstruction/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settingg
        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs['eval']

        # settings for loading an existing experiment
        if (kwargs['is_continue'] or self.eval) and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue'] or self.eval
            
        
        self.exps_folder_name = 'exps'
        self.logs_folder_name = 'logs'
        self.logdir = utils.concat_home_dir(os.path.join(self.home_dir, self.logs_folder_name))
        
        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))
        utils.mkdir_ifnotexists(self.logdir)
        
        self.input_file = self.conf.get_string('train.input_path')
        self.input_file_name = self.input_file.split('/')[-1][:-4]
        
        self.data, self.scale, self.center = utils.load_point_cloud_by_file_extension(self.input_file)
        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)
            
        if is_continue and not(kwargs['otherdir']):
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)
        utils.mkdir_ifnotexists(self.cur_exp_dir)
        self.cur_log_dir = os.path.join(self.logdir, self.timestamp)
        utils.mkdir_ifnotexists(self.cur_log_dir)
            
        sigma_set = []
        ptree = cKDTree(self.data)
        
        for p in np.array_split(self.data, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(np.array(d[0][:, -1]))
        
        sigmas = np.concatenate(sigma_set)
        
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()
            
        self.learning_rate = self.conf.get_string('train.learning_rate_schedule')
        self.network_setting= self.conf.get_string('network')
        
        utils.save_configs(self.cur_exp_dir, is_continue = kwargs['is_continue'], otherdir=kwargs['otherdir'] ,input_path=self.input_file, lr=self.learning_rate, network_setting=self.network_setting, 
        regularizer_type=self.regularizer_type, regularizer_coord=self.regularizer_coord, auggrad=self.auggrad,
        points_batch=kwargs['points_batch'], epsilon = self.epsilon)
        
        self.plots_dir = os.path.join(self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs['nepochs']

        self.points_batch = kwargs['points_batch']

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma,
                                                                                                self.local_sigma)
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')
        
        # use normals if data has  normals and normals_lambda is positive
        self.with_normals = self.normals_lambda > 0 and self.data.shape[-1] >= 6

        self.d_in = self.conf.get_int('train.d_in')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))

        
        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))

        self.weight_decay = self.conf.get_float('train.weight_decay')
        self.checkpoint = kwargs['checkpoint']
        self.startepoch = 0

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])
        # if continue load checkpoints
        
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(old_checkpnts_dir)
            
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(self.checkpoint) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(self.checkpoint) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []
        
        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )
    
        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--points_batch', type=int, default=16384, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='setup.conf')
    parser.add_argument('--expname', type=str, default='single_shape')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='0', type=str)
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--regularizer_type', type=str, default = 'curl', help='none/curl')
    parser.add_argument('--auggrad', type=bool, default = True)
    parser.add_argument('--regularizer_coord', type=float, nargs='+',default = [0.1, 0.0001, 0.0005, 0.1] )
    parser.add_argument('--otherdir', default = False, action='store_true')
    parser.add_argument('--epsilon', type= float, default = 0.1) # In the paper, we report results from experiments with epsilon=1. However, after the submission, we found that epsilon=0.1 generally works better for various geometries.

    args = parser.parse_args()

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu
        
    trainrunner = ReconstructionRunner(
            conf=args.conf,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=gpu,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            eval=args.eval,
            regularizer_type=args.regularizer_type,
            regularizer_coord=args.regularizer_coord,
            auggrad= args.auggrad,
            otherdir = args.otherdir,
            epsilon = args.epsilon,
    )
    
    trainrunner.run()
