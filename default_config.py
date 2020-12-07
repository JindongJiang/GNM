from yacs.config import CfgNode as CN

_C = CN()

_C.exp_name = ''
_C.data_dir = ''
_C.summary_dir = ''
_C.model_dir = ''
_C.last_ckpt = ''


_C.data = CN()
_C.data.img_w = 128
_C.data.img_h = 128
_C.data.inp_channel = 3
_C.data.blender_dir_list_train = []
_C.data.blender_dir_list_test = []
_C.data.dataset = 'mnist'

_C.z = CN()
_C.z.z_global_dim = 32
_C.z.z_what_dim = 10
_C.z.z_where_scale_dim = 2
_C.z.z_where_shift_dim = 2
_C.z.z_where_dim = _C.z.z_where_scale_dim + _C.z.z_where_shift_dim
_C.z.z_pres_dim = 1
_C.z.z_depth_dim = 1
_C.z.z_local_dim = 64
_C.z.z_bg_dim = 10

_C.arch = CN()
_C.arch.glimpse_size = 32
_C.arch.num_cell = 4
_C.arch.phase_overlap = True
_C.arch.phase_background = False
_C.arch.img_enc_dim = 64

_C.arch.conv = CN()
_C.arch.conv.img_encoder_filters = [16, 16, 32, 32, 64, 64, 64, 64, _C.arch.img_enc_dim]
_C.arch.conv.img_encoder_groups = [1, 1, 1, 1, 1, 1, 1, 1, 1]
_C.arch.conv.img_encoder_strides = [2, 1, 2, 1, 2, 1, 2, 1, 2]
_C.arch.conv.img_encoder_kernel_sizes = [4, 3, 4, 3, 4, 3, 4, 3, 4]

_C.arch.deconv = CN()
_C.arch.deconv.p_global_decoder_filters = [128, 128, _C.arch.img_enc_dim]
_C.arch.deconv.p_global_decoder_kernel_sizes = [1, 1, 1]
_C.arch.deconv.p_global_decoder_upscales = [2, 1, 2]
_C.arch.deconv.p_global_decoder_groups = [1, 1, 1]
_C.arch.p_global_decoder_type = 'MLP'
_C.arch.mlp = CN()
_C.arch.mlp.p_global_decoder_filters = [512, 512, _C.arch.img_enc_dim * _C.arch.num_cell ** 2]

_C.arch.mlp.q_global_encoder_filters = [512, 512, _C.z.z_global_dim * 2]
_C.arch.mlp.p_global_encoder_filters = [512, 512, _C.z.z_global_dim * 2]

_C.arch.pwdw = CN()
_C.arch.pwdw.pwdw_filters = [64, 64]
_C.arch.pwdw.pwdw_kernel_sizes = [1, 1]
_C.arch.pwdw.pwdw_strides = [1, 1]
_C.arch.pwdw.pwdw_groups = [1, 1]

if _C.data.inp_channel == 1 or not _C.arch.phase_overlap:
    _C.arch.conv.p_what_decoder_filters = [64, 64, 32, 16, 8, _C.data.inp_channel]
else:
    _C.arch.conv.p_what_decoder_filters = [64, 64, 32, 16, 8, _C.data.inp_channel + 1]

_C.arch.conv.p_what_decoder_kernel_sizes = [1, 1, 1, 1, 3, 3]
_C.arch.conv.p_what_decoder_upscales = [4, 2, 2, 2, 1, 1]
_C.arch.conv.p_what_decoder_groups = [1, 1, 1, 1, 1, 0]


_C.arch.draw_step = 3
_C.arch.structdraw = CN()
_C.arch.structdraw.kernel_size = 1
# _C.arch.structdraw.img_encoder_filters = [512, 512, 128]
_C.arch.structdraw.rnn_decoder_hid_dim = 64
_C.arch.structdraw.rnn_encoder_hid_dim = 64

_C.arch.structdraw.hid_to_dec_filters = [_C.arch.structdraw.rnn_decoder_hid_dim]
_C.arch.structdraw.hid_to_dec_kernel_sizes = [3]
_C.arch.structdraw.hid_to_dec_strides = [1]
_C.arch.structdraw.hid_to_dec_groups = [1]

assert _C.arch.img_enc_dim == _C.arch.structdraw.hid_to_dec_filters[-1] == _C.arch.structdraw.rnn_decoder_hid_dim

_C.arch.phase_graph_net_on_global_decoder = False
_C.arch.phase_graph_net_on_global_encoder = False


# _C.arch.p_bg_decoder_type = 'CONV'
_C.arch.conv.p_bg_decoder_filters = [128, 64, 32, 16, 8, _C.data.inp_channel]
_C.arch.conv.p_bg_decoder_kernel_sizes = [1, 1, 1, 1, 1, 3]
_C.arch.conv.p_bg_decoder_upscales = [4, 2, 4, 2, 2, 1]
_C.arch.conv.p_bg_decoder_groups = [1, 1, 1, 1, 1, 1]

_C.arch.mlp.p_bg_generator_filters = [128, 64, _C.z.z_bg_dim * 2]
_C.arch.mlp.q_bg_encoder_filters = [512, 256, _C.z.z_bg_dim * 2]

_C.log = CN()
_C.log.num_summary_img = 15
_C.log.num_img_per_row = 5
_C.log.save_epoch_freq = 10
_C.log.print_step_freq = 2000
_C.log.num_sample = 50
_C.log.compute_nll_freq = 20
_C.log.phase_nll = False
_C.log.nll_num_sample = 30

_C.const = CN()
_C.const.pres_logit_scale = 8.8
_C.const.scale_mean = -2.2
_C.const.scale_std = 0.1
_C.const.ratio_mean = 0
_C.const.ratio_std = 0.3
_C.const.shift_std = 1
_C.const.eps = 1e-15
_C.const.likelihood_sigma = 0.3
_C.const.bg_likelihood_sigma = 0.3

_C.train = CN()
_C.train.start_epoch = 0
_C.train.epoch = 600
_C.train.batch_size = 32
_C.train.lr = 1e-4
_C.train.cp = 1.0
_C.train.beta_pres_anneal_start_step = 1000
_C.train.beta_pres_anneal_end_step = 0
_C.train.beta_pres_anneal_start_value = 1.
_C.train.beta_pres_anneal_end_value = 0.
_C.train.beta_where_anneal_start_step = 1000
_C.train.beta_where_anneal_end_step = 0
_C.train.beta_where_anneal_start_value = 1.
_C.train.beta_where_anneal_end_value = 0.
_C.train.beta_what_anneal_start_step = 1000
_C.train.beta_what_anneal_end_step = 0
_C.train.beta_what_anneal_start_value = 1.
_C.train.beta_what_anneal_end_value = 0.
_C.train.beta_depth_anneal_start_step = 1000
_C.train.beta_depth_anneal_end_step = 0
_C.train.beta_depth_anneal_start_value = 1.
_C.train.beta_depth_anneal_end_value = 0.
_C.train.beta_global_anneal_start_step = 0
_C.train.beta_global_anneal_end_step = 100000
_C.train.beta_global_anneal_start_value = 0.
_C.train.beta_global_anneal_end_value = 1.
_C.train.beta_bg_anneal_start_step = 1000
_C.train.beta_bg_anneal_end_step = 0
_C.train.beta_bg_anneal_start_value = 1.
_C.train.beta_bg_anneal_end_value = 0.

_C.train.beta_aux_pres_anneal_start_step = 1000
_C.train.beta_aux_pres_anneal_end_step = 0
_C.train.beta_aux_pres_anneal_start_value = 1.
_C.train.beta_aux_pres_anneal_end_value = 0.
_C.train.beta_aux_where_anneal_start_step = 1000
_C.train.beta_aux_where_anneal_end_step = 0
_C.train.beta_aux_where_anneal_start_value = 1.
_C.train.beta_aux_where_anneal_end_value = 0.
_C.train.beta_aux_what_anneal_start_step = 1000
_C.train.beta_aux_what_anneal_end_step = 0
_C.train.beta_aux_what_anneal_start_value = 1.
_C.train.beta_aux_what_anneal_end_value = 0.
_C.train.beta_aux_depth_anneal_start_step = 1000
_C.train.beta_aux_depth_anneal_end_step = 0
_C.train.beta_aux_depth_anneal_start_value = 1.
_C.train.beta_aux_depth_anneal_end_value = 0.
_C.train.beta_aux_global_anneal_start_step = 0
_C.train.beta_aux_global_anneal_end_step = 100000
_C.train.beta_aux_global_anneal_start_value = 0.
_C.train.beta_aux_global_anneal_end_value = 1.
_C.train.beta_aux_bg_anneal_start_step = 1000
_C.train.beta_aux_bg_anneal_end_step = 0
_C.train.beta_aux_bg_anneal_start_value = 1.
_C.train.beta_aux_bg_anneal_end_value = 0.

_C.train.tau_pres_anneal_start_step = 1000
_C.train.tau_pres_anneal_end_step = 20000
_C.train.tau_pres_anneal_start_value = 1.
_C.train.tau_pres_anneal_end_value = 0.5
_C.train.p_pres_anneal_start_step = 0
_C.train.p_pres_anneal_end_step = 4000
_C.train.p_pres_anneal_start_value = 1e-1
_C.train.p_pres_anneal_end_value = 1e-3

_C.train.aux_p_scale_anneal_start_step = 0
_C.train.aux_p_scale_anneal_end_step = 0
_C.train.aux_p_scale_anneal_start_value = _C.const.scale_mean
_C.train.aux_p_scale_anneal_end_value = _C.const.scale_mean

_C.train.phase_bg_alpha_curriculum = False
_C.train.bg_alpha_curriculum_period = [0, 1000]
_C.train.bg_alpha_curriculum_value = 1.

_C.train.seed = 666

def get_cfg_defaults():
    return _C.clone()

