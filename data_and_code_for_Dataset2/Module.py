import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layer import *
from torch.autograd import Variable



def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=1234)


class MS_CAM(nn.Module):
    def __init__(self, channels=2, r=2):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class ResidualBlock(nn.Module):
    def __init__(self, feature_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feature_dim, feature_dim)
    def forward(self, x):
        identity = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += identity
        out = self.relu(out)
        return out


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])
    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1


class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha = 0.5):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)
        self.hgcn2 = HGCN(in_size, hid_list)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.8
        self.alpha = alpha
    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class HGCMLDA(nn.Module):
    def __init__(self, lnc_num, dis_num, hidd_list, num_proj_hidden, hyperpm):
        super(HGCMLDA, self).__init__()
        self.CL_HGCN_lnc = CL_HGCN(lnc_num, hidd_list,num_proj_hidden)
        self.CL_HGCN_dis = CL_HGCN(dis_num, hidd_list,num_proj_hidden)
        self.disease_encoders1 = disease_encoder(lnc_num, 192, 0.3)
        self.lncRNA_encoders1 = lncRNA_encoder(dis_num, 192, 0.3)
        self.decoder2 = decoder2(act=lambda x: x)
        self._enc_mu_disease = disease_encoder(192, 128, 0.3, act=lambda x: x)
        self._enc_log_sigma_disease = disease_encoder(192, 128, 0.3, act=lambda x: x)
        self._enc_mu_lncRNA = lncRNA_encoder(192, 128, 0.3, act=lambda x: x)
        self._enc_log_sigma_lncRNA = lncRNA_encoder(192, 128, 0.3, act=lambda x: x)
        self.att_lnc = MS_CAM()
        self.att_dis = MS_CAM()
        self.res_lnc=ResidualBlock(128)
        self.res_dis=ResidualBlock(128)
        #self.cnn_dis = nn.Conv1d(in_channels=2,
        #                       out_channels=80,
        #                       kernel_size=128,
        #                       stride=1,
        #                       bias=True)
        #self.cnn_lnc = nn.Conv1d(in_channels=2,
        #                       out_channels=80,
        #                       kernel_size=128,
        #                       stride=1,
        #                       bias=True)
        self.linear_L_3 = nn.Linear(256, 64)
        self.linear_D_3 = nn.Linear(256, 64)
        self.linear_L_1 = nn.Linear(128, 128)
        self.linear_D_1 = nn.Linear(128, 128)
        self.linear_L_2 = nn.Linear(128, 128)
        self.linear_D_2 = nn.Linear(128, 128)

    def sample_latent(self, z_disease, z_lncRNA):
        self.z_disease_mean = self._enc_mu_disease(z_disease)
        self.z_disease_log_std = self._enc_log_sigma_disease(z_disease)
        self.z_disease_std = torch.exp(self.z_disease_log_std)
        z_disease_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_disease_std.size())).double()
        self.z_disease_std_ = z_disease_std_.cuda(0)
        self.z_disease_ = self.z_disease_mean + self.z_disease_std.mul(Variable(self.z_disease_std_, requires_grad=True))
        self.z_lncRNA_mean = self._enc_mu_lncRNA(z_lncRNA)
        self.z_lncRNA_log_std = self._enc_log_sigma_lncRNA(z_lncRNA)
        self.z_lncRNA_std = torch.exp(self.z_lncRNA_log_std)
        z_lncRNA_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_lncRNA_std.size())).double()
        self.z_lncRNA_std_ = z_lncRNA_std_.cuda(0)
        self.z_lncRNA_ = self.z_lncRNA_mean + self.z_lncRNA_std.mul(Variable(self.z_lncRNA_std_, requires_grad=True))
        if self.training:
            return self.z_disease_, self.z_lncRNA_
        else:
            return self.z_disease_mean, self.z_lncRNA_mean

    def forward(self, concat_lnc_tensor, concat_dis_tensor, G_lnc_Kn, G_lnc_Gm, G_dis_Kn, G_dis_Gm,LD,DL):
        z_disease_encoder = self.disease_encoders1(DL)
        z_lncRNA_encoder = self.lncRNA_encoders1(LD)
        self.z_disease_s, self.z_lncRNA_s = self.sample_latent(z_disease_encoder, z_lncRNA_encoder)
        z_disease = self.z_disease_s
        z_lncRNA=self.z_lncRNA_s
        reconstruction_VAE = self.decoder2(z_lncRNA,z_disease)
        reconstruction_VAE=torch.relu(reconstruction_VAE)
        lnc_embedded = concat_lnc_tensor
        dis_embedded = concat_dis_tensor
        lnc_feature1, lnc_feature2, lnc_cl_loss = self.CL_HGCN_lnc(lnc_embedded, G_lnc_Kn, lnc_embedded, G_lnc_Gm)
        lnc_feature1=torch.unsqueeze(lnc_feature1, dim=0)
        lnc_feature2=torch.unsqueeze(lnc_feature2, dim=0)
        lnc_att = torch.cat((lnc_feature1, lnc_feature2), 0)
        lnc_att = lnc_att.unsqueeze(0)
        lnc_att = lnc_att.permute(2, 1, 0, 3)
        lnc_att = self.att_lnc(lnc_att)
        lnc_att = lnc_att.permute(2, 1, 0, 3)
        lnc_att = torch.squeeze(lnc_att, dim=0)
        lnc_att = (lnc_att[0] + lnc_att[1]) / 2
        lnc_vae=z_lncRNA.to(dtype=torch.float32)
        lnc_vae=torch.relu(self.linear_L_1(lnc_vae))
        lnc_att=torch.relu(self.linear_L_2(lnc_att))
        concat_lnc=torch.cat((lnc_att,lnc_vae),1)
        dis_feature1, dis_feature2, dis_cl_loss = self.CL_HGCN_dis(dis_embedded, G_dis_Kn, dis_embedded, G_dis_Gm)
        dis_feature1=torch.unsqueeze(dis_feature1, dim=0)
        dis_feature2=torch.unsqueeze(dis_feature2, dim=0)
        dis_att = torch.cat((dis_feature1, dis_feature2), 0)
        dis_att = dis_att.unsqueeze(0)
        dis_att = dis_att.permute(2, 1, 0, 3)
        dis_att = self.att_dis(dis_att)
        dis_att = dis_att.permute(2, 1, 0, 3)
        dis_att = torch.squeeze(dis_att, dim=0)
        dis_att = (dis_att[0] + dis_att[1]) / 2
        dis_vae=z_disease.to(dtype=torch.float32)
        dis_vae=torch.relu(self.linear_D_1(dis_vae))
        dis_att=torch.relu(self.linear_D_2(dis_att))
        concat_dis=torch.cat((dis_att,dis_vae),1)
        lnc_after_fc=torch.relu(self.linear_L_3(concat_lnc))
        dis_after_fc=torch.relu(self.linear_D_3(concat_dis))
        score=lnc_after_fc.mm(dis_after_fc.t())
        return score, lnc_cl_loss, dis_cl_loss,reconstruction_VAE



