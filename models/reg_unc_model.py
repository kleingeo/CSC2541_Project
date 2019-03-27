import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from collections import OrderedDict

from . import loss

import torch.nn as nn

# train only generator

# Define model using network that removes tanh from final layer
class RegressionUncertaintyNetModel(BaseModel):
    def name(self):
        return 'RegressionUncertaintyNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_L1unc', 'G_unc']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
#        self.visual_names = ['real_A', 'fake_B', 'real_B', 'var_B']
        self.visual_names = []

        # specify images to display individual channels for.
        # overload base_model.get_current_visuals for this.
        self.visual_names_channels = ['real_A', 'fake_B', 'real_B', 'var_B', 'err_B']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1unc = loss.L1_uncertainty_loss(N=7.0)
            self.criterionUnc = loss.log_variance_loss(N=7.0)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)

        for name in self.visual_names_channels:
            if isinstance(name, str):
                img = getattr(self, name)

                nc = img.shape[1]
                for channel in range(nc):
                    x = img[:, channel, :, :]
                    x = x.unsqueeze(1)
                    visual_ret['{:s}{:01d}'.format(name, channel)] = x

        return visual_ret

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        x = self.netG(self.real_A)
        self.fake_B = x[:, 0:self.opt.output_nc, ...]
        self.unc_B = x[:, self.opt.output_nc:self.opt.output_nc*2, ...]

        #self.fake_B = self.netG(self.real_A)
        # calculate variance from log-variance

        N = 7.0
        # variance is between [exp(-N), exp(+N)]
        # -- scale to [exp(-N) - 1.0, exp(+N)-1.0]
        # -- when displayed, this clips to [-1, 1] -- black=0, white=2
        self.var_B = torch.exp(N * self.unc_B).detach() - 1.0
        self.err_B = (self.fake_B - self.real_B).detach()

    def backward_G(self):
        self.loss_G_L1unc = self.criterionL1unc(self.fake_B, self.unc_B, self.real_B)
        self.loss_G_unc = self.criterionUnc(self.unc_B)
        self.loss_G = self.loss_G_L1unc + self.loss_G_unc
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
