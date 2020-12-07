import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Tuple
from submodule import StackConvNorm, StackSubPixelNorm, \
    StackMLP, ConvLSTMCell
from torch.distributions import RelaxedBernoulli, Normal


class ImgEncoder(nn.Module):

    def __init__(self, args: Any):
        super(ImgEncoder, self).__init__()

        self.args = args
        self.enc = StackConvNorm(
            self.args.data.inp_channel,
            self.args.arch.conv.img_encoder_filters,
            self.args.arch.conv.img_encoder_kernel_sizes,
            self.args.arch.conv.img_encoder_strides,
            self.args.arch.conv.img_encoder_groups,
            norm_act_final=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)

        return x


class LocalLatentDecoder(nn.Module):

    def __init__(self, args: Any):
        super(LocalLatentDecoder, self).__init__()
        self.args = args

        pwdw_net_inp_dim = self.args.arch.img_enc_dim


        self.pwdw_net = StackConvNorm(
            pwdw_net_inp_dim,
            self.args.arch.pwdw.pwdw_filters,
            self.args.arch.pwdw.pwdw_kernel_sizes,
            self.args.arch.pwdw.pwdw_strides,
            self.args.arch.pwdw.pwdw_groups,
            norm_act_final=True
        )

        self.q_depth_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_depth_dim * 2, 1)
        self.q_where_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_where_dim * 2, 1)
        self.q_what_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_what_dim * 2, 1)
        self.q_pres_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_pres_dim, 1)

        torch.nn.init.uniform_(self.q_where_net.weight.data, -0.01, 0.01)
        # scale
        torch.nn.init.constant_(self.q_where_net.bias.data[0], -1.)
        # ratio, x, y, std
        torch.nn.init.constant_(self.q_where_net.bias.data[1:], 0)

    def forward(self, img_enc: torch.Tensor, ss_p_z: List = None) -> List:
        """

        :param img_enc: (bs, dim, 4, 4)
        :param global_dec: (bs, dim, 4, 4)
        :return:
        """

        if ss_p_z is not None:
            p_pres_logits, p_where_mean, p_where_std, p_depth_mean, \
            p_depth_std, p_what_mean, p_what_std = ss_p_z

        pwdw_inp = img_enc

        pwdw_ss = self.pwdw_net(pwdw_inp)

        q_pres_logits = self.q_pres_net(pwdw_ss).tanh() * self.args.const.pres_logit_scale

        # q_where_mean, q_where_std: (bs, dim, num_cell, num_cell)
        q_where_mean, q_where_std = \
            self.q_where_net(pwdw_ss).chunk(2, 1)
        q_where_std = F.softplus(q_where_std)

        # q_depth_mean, q_depth_std: (bs, dim, num_cell, num_cell)
        q_depth_mean, q_depth_std = \
            self.q_depth_net(pwdw_ss).chunk(2, 1)
        q_depth_std = F.softplus(q_depth_std)

        q_what_mean, q_what_std = \
            self.q_what_net(pwdw_ss).chunk(2, 1)
        q_what_std = F.softplus(q_what_std)

        ss = [
            q_pres_logits, q_where_mean, q_where_std,
            q_depth_mean, q_depth_std, q_what_mean, q_what_std
        ]

        return ss


class LocalLatentGenerator(nn.Module):

    def __init__(self, args: Any):
        super(LocalLatentGenerator, self).__init__()
        self.args = args

        self.pwdw_net = StackConvNorm(
            self.args.arch.img_enc_dim,
            self.args.arch.pwdw.pwdw_filters,
            self.args.arch.pwdw.pwdw_kernel_sizes,
            self.args.arch.pwdw.pwdw_strides,
            self.args.arch.pwdw.pwdw_groups,
            norm_act_final=True
        )

        self.p_depth_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_depth_dim * 2, 1)
        self.p_where_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_where_dim * 2, 1)
        self.p_what_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_what_dim * 2, 1)
        self.p_pres_net = nn.Conv2d(self.args.arch.pwdw.pwdw_filters[-1], self.args.z.z_pres_dim, 1)

        torch.nn.init.uniform_(self.p_where_net.weight.data, -0.01, 0.01)
        # scale
        torch.nn.init.constant_(self.p_where_net.bias.data[0], -1.)
        # ratio, x, y, std
        torch.nn.init.constant_(self.p_where_net.bias.data[1:], 0)

    def forward(self, global_dec: torch.Tensor) -> List:
        """

        :param x: sample of img_enc (bs, dim, 4, 4)
        :return:
        """

        pwdw_ss = self.pwdw_net(global_dec)

        p_pres_logits = self.p_pres_net(pwdw_ss).tanh() * self.args.const.pres_logit_scale

        # p_where_mean, p_where_std: (bs, dim, num_cell, num_cell)
        p_where_mean, p_where_std = \
            self.p_where_net(pwdw_ss).chunk(2, 1)
        p_where_std = F.softplus(p_where_std)

        # p_depth_mean, p_depth_std: (bs, dim, num_cell, num_cell)
        p_depth_mean, p_depth_std = \
            self.p_depth_net(pwdw_ss).chunk(2, 1)
        p_depth_std = F.softplus(p_depth_std)

        p_what_mean, p_what_std = \
            self.p_what_net(pwdw_ss).chunk(2, 1)
        p_what_std = F.softplus(p_what_std)

        ss = [
            p_pres_logits, p_where_mean, p_where_std,
            p_depth_mean, p_depth_std, p_what_mean, p_what_std
        ]

        return ss


class LocalSampler(nn.Module):

    def __init__(self, args: Any):
        super(LocalSampler, self).__init__()
        self.args = args

        self.z_what_decoder_net = StackSubPixelNorm(
            self.args.z.z_what_dim,
            self.args.arch.conv.p_what_decoder_filters,
            self.args.arch.conv.p_what_decoder_kernel_sizes,
            self.args.arch.conv.p_what_decoder_upscales,
            self.args.arch.conv.p_what_decoder_groups,
            norm_act_final=False
        )

        self.register_buffer('offset', torch.stack(
            torch.meshgrid(torch.arange(args.arch.num_cell).float(),
                           torch.arange(args.arch.num_cell).float())[::-1], dim=0
        ).view(1, 2, args.arch.num_cell, args.arch.num_cell))

    def forward(self, ss: List, phase_use_mode: bool = False) -> Tuple:

        p_pres_logits, p_where_mean, p_where_std, p_depth_mean, \
        p_depth_std, p_what_mean, p_what_std = ss

        if phase_use_mode:
            z_pres = (p_pres_logits > 0).float()
        else:
            z_pres = RelaxedBernoulli(logits=p_pres_logits, temperature=self.args.train.tau_pres).rsample()

        # z_where_scale, z_where_shift: (bs, dim, num_cell, num_cell)
        if phase_use_mode:
            z_where_scale, z_where_shift = p_where_mean.chunk(2, 1)
        else:
            z_where_scale, z_where_shift = \
                Normal(p_where_mean, p_where_std).rsample().chunk(2, 1)

        # z_where_origin: (bs, dim, num_cell, num_cell)
        z_where_origin = \
            torch.cat([z_where_scale.detach(), z_where_shift.detach()], dim=1)

        z_where_shift = \
            (2. / self.args.arch.num_cell) * \
            (self.offset + 0.5 + torch.tanh(z_where_shift)) - 1.

        scale, ratio = z_where_scale.chunk(2, 1)
        scale = scale.sigmoid()
        ratio = torch.exp(ratio)
        ratio_sqrt = ratio.sqrt()
        z_where_scale = torch.cat([scale / ratio_sqrt, scale * ratio_sqrt], dim=1)
        # z_where: (bs, dim, num_cell, num_cell)
        z_where = torch.cat([z_where_scale, z_where_shift], dim=1)

        if phase_use_mode:
            z_depth = p_depth_mean
            z_what = p_what_mean
        else:
            z_depth = Normal(p_depth_mean, p_depth_std).rsample()
            z_what = Normal(p_what_mean, p_what_std).rsample()

        z_what_reshape = z_what.permute(0, 2, 3, 1).reshape(-1, self.args.z.z_what_dim). \
            view(-1, self.args.z.z_what_dim, 1, 1)

        if self.args.data.inp_channel == 1 or not self.args.arch.phase_overlap:
            o = self.z_what_decoder_net(z_what_reshape)
            o = o.sigmoid()
            a = o.new_ones(o.size())
        elif self.args.arch.phase_overlap:
            o, a = self.z_what_decoder_net(z_what_reshape).split([self.args.data.inp_channel, 1], dim=1)
            o, a = o.sigmoid(), a.sigmoid()
        else:
            raise NotImplemented

        lv = [z_pres, z_where, z_depth, z_what, z_where_origin]
        pa = [o, a]

        return pa, lv


class StructDRAW(nn.Module):

    def __init__(self, args):
        super(StructDRAW, self).__init__()
        self.args = args

        self.p_global_decoder_net = StackMLP(
            self.args.z.z_global_dim,
            self.args.arch.mlp.p_global_decoder_filters,
            norm_act_final=True
        )

        rnn_enc_inp_dim = self.args.arch.img_enc_dim * 2 + \
                          self.args.arch.structdraw.rnn_decoder_hid_dim

        rnn_dec_inp_dim = self.args.arch.mlp.p_global_decoder_filters[-1] // \
                          (self.args.arch.num_cell ** 2)

        rnn_dec_inp_dim += self.args.arch.structdraw.hid_to_dec_filters[-1]

        self.rnn_enc = ConvLSTMCell(
            input_dim=rnn_enc_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_encoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.rnn_dec = ConvLSTMCell(
            input_dim=rnn_dec_inp_dim,
            hidden_dim=self.args.arch.structdraw.rnn_decoder_hid_dim,
            kernel_size=self.args.arch.structdraw.kernel_size,
            num_cell=self.args.arch.num_cell
        )

        self.p_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.mlp.p_global_encoder_filters,
            norm_act_final=False
        )

        self.q_global_net = StackMLP(
            self.args.arch.num_cell ** 2 * self.args.arch.structdraw.rnn_encoder_hid_dim,
            self.args.arch.mlp.q_global_encoder_filters,
            norm_act_final=False
        )

        self.hid_to_dec_net = StackConvNorm(
            self.args.arch.structdraw.rnn_decoder_hid_dim,
            self.args.arch.structdraw.hid_to_dec_filters,
            self.args.arch.structdraw.hid_to_dec_kernel_sizes,
            self.args.arch.structdraw.hid_to_dec_strides,
            self.args.arch.structdraw.hid_to_dec_groups,
            norm_act_final=False
        )

        self.register_buffer('dec_step_0', torch.zeros(1, self.args.arch.structdraw.hid_to_dec_filters[-1],
                                                       self.args.arch.num_cell, self.args.arch.num_cell))

    def forward(self, x: torch.Tensor, phase_generation: bool = False,
                generation_from_step: Any = None, z_global_predefine: Any = None) -> Tuple:
        """
        :param x: (bs, dim, num_cell, num_cell) of (bs, dim, img_h, img_w)
        :return:
        """

        bs = x.size(0)

        h_enc, c_enc = self.rnn_enc.init_hidden(bs)
        h_dec, c_dec = self.rnn_dec.init_hidden(bs)

        p_global_mean_list = []
        p_global_std_list = []
        q_global_mean_list = []
        q_global_std_list = []
        z_global_list = []

        dec_step = self.dec_step_0.expand(bs, -1, -1, -1)

        for i in range(self.args.arch.draw_step):

            p_global_mean_step, p_global_std_step = \
                self.p_global_net(h_dec.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)
            p_global_std_step = F.softplus(p_global_std_step)

            if phase_generation or (generation_from_step is not None and i >= generation_from_step):

                q_global_mean_step = x.new_empty(bs, self.args.z.z_global_dim)
                q_global_std_step = x.new_empty(bs, self.args.z.z_global_dim)

                if z_global_predefine is None or z_global_predefine.size(1) <= i:
                    z_global_step = Normal(p_global_mean_step, p_global_std_step).rsample()
                else:
                    z_global_step = z_global_predefine.view(bs, -1, self.args.z.z_global_dim)[:, i]

            else:

                if i == 0:
                    rnn_encoder_inp = torch.cat([x, x, h_dec], dim=1)
                else:
                    rnn_encoder_inp = torch.cat([x, x - dec_step, h_dec], dim=1)

                h_enc, c_enc = self.rnn_enc(rnn_encoder_inp, [h_enc, c_enc])

                q_global_mean_step, q_global_std_step = \
                    self.q_global_net(h_enc.permute(0, 2, 3, 1).reshape(bs, -1)).chunk(2, -1)

                q_global_std_step = F.softplus(q_global_std_step)
                z_global_step = Normal(q_global_mean_step, q_global_std_step).rsample()

            rnn_decoder_inp = self.p_global_decoder_net(z_global_step). \
                reshape(bs, -1, self.args.arch.num_cell, self.args.arch.num_cell)

            rnn_decoder_inp = torch.cat([rnn_decoder_inp, dec_step], dim=1)

            h_dec, c_dec = self.rnn_dec(rnn_decoder_inp, [h_dec, c_dec])

            dec_step = dec_step + self.hid_to_dec_net(h_dec)

            # (bs, dim)
            p_global_mean_list.append(p_global_mean_step)
            p_global_std_list.append(p_global_std_step)
            q_global_mean_list.append(q_global_mean_step)
            q_global_std_list.append(q_global_std_step)
            z_global_list.append(z_global_step)

        global_dec = dec_step

        # (bs, steps, dim, 1, 1)
        p_global_mean_all = torch.stack(p_global_mean_list, 1)[:, :, :, None, None]
        p_global_std_all = torch.stack(p_global_std_list, 1)[:, :, :, None, None]
        q_global_mean_all = torch.stack(q_global_mean_list, 1)[:, :, :, None, None]
        q_global_std_all = torch.stack(q_global_std_list, 1)[:, :, :, None, None]
        z_global_all = torch.stack(z_global_list, 1)[:, :, :, None, None]

        pa = [global_dec]
        lv = [z_global_all]
        ss = [p_global_mean_all, p_global_std_all, q_global_mean_all, q_global_std_all]

        return pa, lv, ss

class BgEncoder(nn.Module):

    def __init__(self, args):
        super(BgEncoder, self).__init__()
        self.args = args

        self.p_bg_encoder = StackMLP(
            self.args.arch.img_enc_dim * self.args.arch.num_cell ** 2,
            self.args.arch.mlp.q_bg_encoder_filters,
            norm_act_final=False
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        """
        :param x: (bs, dim, img_h, img_w) or (bs, dim, num_cell, num_cell)
        :return:
        """
        bs = x.size(0)
        q_bg_mean, q_bg_std = self.p_bg_encoder(x.view(bs, -1)).chunk(2, 1)
        q_bg_mean = q_bg_mean.view(bs, -1, 1, 1)
        q_bg_std = q_bg_std.view(bs, -1, 1, 1)

        q_bg_std = F.softplus(q_bg_std)

        z_bg = Normal(q_bg_mean, q_bg_std).rsample()

        lv = [z_bg]

        ss = [q_bg_mean, q_bg_std]

        return lv, ss


class BgGenerator(nn.Module):

    def __init__(self, args):
        super(BgGenerator, self).__init__()
        self.args = args

        inp_dim = self.args.z.z_global_dim * self.args.arch.draw_step

        self.p_bg_generator = StackMLP(
            inp_dim,
            self.args.arch.mlp.p_bg_generator_filters,
            norm_act_final=False
        )

    def forward(self, z_global_all: torch.Tensor, phase_use_mode: bool = False) -> Tuple:
        """
        :param x: (bs, step, dim, 1, 1)
        :return:
        """
        bs = z_global_all.size(0)

        bg_generator_inp = z_global_all

        p_bg_mean, p_bg_std = self.p_bg_generator(bg_generator_inp.view(bs, -1)).chunk(2, 1)
        p_bg_std = F.softplus(p_bg_std)

        p_bg_mean = p_bg_mean.view(bs, -1, 1, 1)
        p_bg_std = p_bg_std.view(bs, -1, 1, 1)

        if phase_use_mode:
            z_bg = p_bg_mean
        else:
            z_bg = Normal(p_bg_mean, p_bg_std).rsample()

        lv = [z_bg]

        ss = [p_bg_mean, p_bg_std]

        return lv, ss


class BgDecoder(nn.Module):

    def __init__(self, args):
        super(BgDecoder, self).__init__()
        self.args = args

        self.p_bg_decoder = StackSubPixelNorm(
            self.args.z.z_bg_dim,
            self.args.arch.conv.p_bg_decoder_filters,
            self.args.arch.conv.p_bg_decoder_kernel_sizes,
            self.args.arch.conv.p_bg_decoder_upscales,
            self.args.arch.conv.p_bg_decoder_groups,
            norm_act_final=False
        )

    def forward(self, z_bg: torch.Tensor) -> List:
        """
        :param x: (bs, dim, 1, 1)
        :return:
        """
        bs = z_bg.size(0)

        bg = self.p_bg_decoder(z_bg).sigmoid()

        pa = [bg]

        return pa
