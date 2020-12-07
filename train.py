import os
import time
import torch
import math
import yaml

import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch import nn
from data import MultiMNIST, Blender
from utils import save_ckpt, load_ckpt, print_schedule, \
    visualize, linear_schedule, log_mean_exp

from config import get_config

from torch.utils.tensorboard import SummaryWriter

from model import GNM


def main():
    # torch.autograd.set_detect_anomaly(True)

    torch.backends.cudnn.benchmark = True

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    args = get_config()[0]

    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    torch.cuda.manual_seed_all(args.train.seed)
    np.random.seed(args.train.seed)

    model_dir = os.path.join(args.model_dir, args.exp_name)
    summary_dir = os.path.join(args.summary_dir, args.exp_name)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(args.seed)
    args.train.num_gpu = torch.cuda.device_count()
    with open(os.path.join(summary_dir, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)
    if args.data.dataset == 'mnist':
        train_data = MultiMNIST(args, mode='train')
        test_data = MultiMNIST(args, mode='test')
        val_data = MultiMNIST(args, mode='val')
    elif args.data.dataset == 'blender':
        train_data = Blender(args, mode='train')
        test_data = Blender(args, mode='test')
        val_data = Blender(args, mode='val')
    else:
        raise NotImplemented

    train_loader = DataLoader(
        train_data, batch_size=args.train.batch_size, shuffle=True, drop_last=True, num_workers=6)
    num_train = len(train_data)

    test_loader = DataLoader(
        test_data, batch_size=args.train.batch_size * 4, shuffle=False, drop_last=True, num_workers=6)
    num_test = len(test_data)

    val_loader = DataLoader(
        val_data, batch_size=args.train.batch_size * 4, shuffle=False, drop_last=True, num_workers=6)
    num_val = len(val_data)

    model = GNM(args)
    model.to(device)
    num_gpu = 1
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        num_gpu = torch.cuda.device_count()
        model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.train.lr)

    global_step = 0
    if args.last_ckpt:
        global_step, args.train.start_epoch = \
            load_ckpt(model, optimizer, args.last_ckpt, device)

    args.train.global_step = global_step
    args.log.phase_log = False

    writer = SummaryWriter(summary_dir)

    end_time = time.time()

    for epoch in range(int(args.train.start_epoch), args.train.epoch):

        local_count = 0
        last_count = 0
        for batch_idx, sample in enumerate(train_loader):

            imgs = sample.to(device)

            hyperparam_anneal(args, global_step)

            global_step += 1

            phase_log = global_step % args.log.print_step_freq == 0 or global_step == 1
            args.train.global_step = global_step
            args.log.phase_log = phase_log

            pa_recon, log_like, kl, _, _, _, log = \
                model(imgs)

            aux_kl_pres, aux_kl_where, aux_kl_depth, aux_kl_what, aux_kl_bg, kl_pres, \
            kl_where, kl_depth, kl_what, kl_global_all, kl_bg = kl

            aux_kl_pres_raw = aux_kl_pres.mean(dim=0)
            aux_kl_where_raw = aux_kl_where.mean(dim=0)
            aux_kl_depth_raw = aux_kl_depth.mean(dim=0)
            aux_kl_what_raw = aux_kl_what.mean(dim=0)
            aux_kl_bg_raw = aux_kl_bg.mean(dim=0)
            kl_pres_raw = kl_pres.mean(dim=0)
            kl_where_raw = kl_where.mean(dim=0)
            kl_depth_raw = kl_depth.mean(dim=0)
            kl_what_raw = kl_what.mean(dim=0)
            kl_bg_raw = kl_bg.mean(dim=0)

            log_like = log_like.mean(dim=0)

            aux_kl_pres = aux_kl_pres_raw * args.train.beta_aux_pres
            aux_kl_where = aux_kl_where_raw * args.train.beta_aux_where
            aux_kl_depth = aux_kl_depth_raw * args.train.beta_aux_depth
            aux_kl_what = aux_kl_what_raw * args.train.beta_aux_what
            aux_kl_bg = aux_kl_bg_raw * args.train.beta_aux_bg
            kl_pres = kl_pres_raw * args.train.beta_pres
            kl_where = kl_where_raw * args.train.beta_where
            kl_depth = kl_depth_raw * args.train.beta_depth
            kl_what = kl_what_raw * args.train.beta_what
            kl_bg = kl_bg_raw * args.train.beta_bg

            kl_global_raw = kl_global_all.sum(dim=-1).mean(dim=0)
            kl_global = kl_global_raw * args.train.beta_global

            total_loss = - (log_like - kl_pres - kl_where - kl_depth - kl_what - kl_bg - kl_global -
                            aux_kl_pres - aux_kl_where - aux_kl_depth - aux_kl_what - aux_kl_bg)

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(model.parameters(), args.train.cp)
            optimizer.step()

            local_count += imgs.data.shape[0]
            if phase_log:

                bs = imgs.size(0)

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_schedule(global_step, epoch, local_count, count_inter,
                               num_train, total_loss, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(
                        'param/' + name, param.cpu().detach().numpy(), global_step)
                    if param.grad is not None:
                        writer.add_histogram(
                            'grad/' + name, param.grad.cpu().detach(), global_step)
                        if len(param.size()) != 1:
                            writer.add_scalar(
                                'grad_std/' + name + '.grad', param.grad.cpu().detach().std().item(), global_step)
                        writer.add_scalar(
                            'grad_mean/' + name + '.grad', param.grad.cpu().detach().mean().item(), global_step)

                for key, value in log.items():
                    if value is None:
                        continue

                    if key == 'importance_map_full_res_norm':
                        writer.add_histogram('inside_value/' + key, value[value > 0].cpu().detach().numpy(),
                                             global_step)
                    else:
                        writer.add_histogram('inside_value/' + key, value.cpu().detach().numpy(),
                                             global_step)

                grid_image = make_grid(
                    imgs.cpu().detach()[:args.log.num_summary_img].view(-1, args.data.inp_channel, args.data.img_h,
                                                                        args.data.img_w),
                    args.log.num_img_per_row, normalize=False, pad_value=1)
                writer.add_image('train/1-image', grid_image, global_step)

                grid_image = make_grid(
                    pa_recon[0].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                        view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                    args.log.num_img_per_row, normalize=False, pad_value=1)
                writer.add_image('train/2-reconstruction_overall', grid_image, global_step)

                if args.arch.phase_background:
                    grid_image = make_grid(
                        pa_recon[1].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/3-reconstruction-fg', grid_image, global_step)

                    grid_image = make_grid(
                        pa_recon[2].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, 1, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/4-reconstruction-alpha', grid_image, global_step)

                    grid_image = make_grid(
                        pa_recon[3].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/5-reconstruction-bg', grid_image, global_step)

                bbox = visualize(imgs[:args.log.num_summary_img].cpu(),
                                 log['z_pres'].view(bs, args.arch.num_cell ** 2, -1)[
                                 :args.log.num_summary_img].cpu().detach(),
                                 log['z_where_scale'].view(bs, args.arch.num_cell ** 2, -1)[
                                 :args.log.num_summary_img].cpu().detach(),
                                 log['z_where_shift'].view(bs, args.arch.num_cell ** 2, -1)[
                                 :args.log.num_summary_img].cpu().detach(),
                                 only_bbox=True, phase_only_display_pres=False)

                bbox = bbox.view(args.log.num_summary_img, -1, 3, args.data.img_h,
                                 args.data.img_w).sum(1).clamp(0.0, 1.0)
                bbox_img = imgs[:args.log.num_summary_img].cpu().expand(-1, 3, -1, -1).contiguous()
                bbox_img[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5] = \
                    bbox[bbox.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5]
                grid_image = make_grid(bbox_img, args.log.num_img_per_row, normalize=False, pad_value=1)

                writer.add_image('train/6-bbox', grid_image, global_step)

                bbox_white = visualize(imgs[:args.log.num_summary_img].cpu(),
                                       log['z_pres'].view(bs, args.arch.num_cell ** 2, -1)[
                                       :args.log.num_summary_img].cpu().detach(),
                                       log['z_where_scale'].view(bs, args.arch.num_cell ** 2, -1)[
                                       :args.log.num_summary_img].cpu().detach(),
                                       log['z_where_shift'].view(bs, args.arch.num_cell ** 2, -1)[
                                       :args.log.num_summary_img].cpu().detach(),
                                       only_bbox=True, phase_only_display_pres=True)

                bbox_white = bbox_white.view(args.log.num_summary_img, -1, 3, args.data.img_h,
                                             args.data.img_w).sum(1).clamp(0.0, 1.0)
                bbox_white_img = imgs[:args.log.num_summary_img].cpu().expand(-1, 3, -1, -1).contiguous()
                bbox_white_img[bbox_white.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5] = \
                    bbox_white[bbox_white.sum(dim=1, keepdim=True).expand(-1, 3, -1, -1) > 0.5]
                grid_image = make_grid(bbox_white_img, args.log.num_img_per_row, normalize=False, pad_value=1)

                writer.add_image('train/6a-bbox-white', grid_image, global_step)

                grid_image = make_grid(
                    log['recon_from_q_g'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                        view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                    args.log.num_img_per_row, normalize=False, pad_value=1)
                writer.add_image('train/7-reconstruction_from_q_g', grid_image, global_step)

                if args.arch.phase_background:
                    grid_image = make_grid(
                        log['recon_from_q_g_fg'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/8-recon_from_q_g-fg', grid_image, global_step)

                    grid_image = make_grid(
                        log['recon_from_q_g_alpha'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, 1, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/9-recon_from_q_g-alpha', grid_image, global_step)

                    grid_image = make_grid(
                        log['recon_from_q_g_bg'].cpu().detach()[:args.log.num_summary_img].clamp(0, 1).
                            view(-1, args.data.inp_channel, args.data.img_h, args.data.img_w),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('train/a-background_from_q_g', grid_image, global_step)

                writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step)
                writer.add_scalar('train/log_like', log_like.item(), global_step=global_step)
                writer.add_scalar('train/What_KL', kl_what.item(), global_step=global_step)
                writer.add_scalar('train/bg_KL', kl_bg.item(), global_step=global_step)
                writer.add_scalar('train/Where_KL', kl_where.item(), global_step=global_step)
                writer.add_scalar('train/Pres_KL', kl_pres.item(), global_step=global_step)
                writer.add_scalar('train/Depth_KL', kl_depth.item(), global_step=global_step)
                writer.add_scalar('train/kl_global', kl_global.item(), global_step=global_step)
                writer.add_scalar('train/What_KL_raw', kl_what_raw.item(), global_step=global_step)
                writer.add_scalar('train/bg_KL_raw', kl_bg_raw.item(), global_step=global_step)
                writer.add_scalar('train/Where_KL_raw', kl_where_raw.item(), global_step=global_step)
                writer.add_scalar('train/Pres_KL_raw', kl_pres_raw.item(), global_step=global_step)
                writer.add_scalar('train/Depth_KL_raw', kl_depth_raw.item(), global_step=global_step)

                writer.add_scalar('train/aux_What_KL', aux_kl_what.item(), global_step=global_step)
                writer.add_scalar('train/aux_bg_KL', aux_kl_bg.item(), global_step=global_step)
                writer.add_scalar('train/aux_Where_KL', aux_kl_where.item(), global_step=global_step)
                writer.add_scalar('train/aux_Pres_KL', aux_kl_pres.item(), global_step=global_step)
                writer.add_scalar('train/aux_Depth_KL', aux_kl_depth.item(), global_step=global_step)
                writer.add_scalar('train/aux_What_KL_raw', aux_kl_what_raw.item(), global_step=global_step)
                writer.add_scalar('train/aux_bg_KL_raw', aux_kl_bg_raw.item(), global_step=global_step)
                writer.add_scalar('train/aux_Where_KL_raw', aux_kl_where_raw.item(), global_step=global_step)
                writer.add_scalar('train/aux_Pres_KL_raw', aux_kl_pres_raw.item(), global_step=global_step)
                writer.add_scalar('train/aux_Depth_KL_raw', aux_kl_depth_raw.item(), global_step=global_step)

                writer.add_scalar('train/kl_global_raw', kl_global_raw.item(), global_step=global_step)
                writer.add_scalar('train/tau_pres', args.train.tau_pres, global_step=global_step)
                for i in range(args.arch.draw_step):
                    writer.add_scalar(f'train/kl_global_raw_step_{i}', kl_global_all[:, i].mean().item(),
                                      global_step=global_step)

                writer.add_scalar('train/log_prob_x_given_g',
                                  log['log_prob_x_given_g'].mean(0).item(), global_step=global_step)

                elbo = (log_like.item() - kl_pres_raw.item() - kl_where_raw.item() - kl_depth_raw.item() -
                        kl_what_raw.item() - kl_bg_raw.item() - kl_global_raw.item())

                writer.add_scalar('train/elbo', elbo, global_step=global_step)

                ######################################## generation ########################################

                with torch.no_grad():
                    model.eval()
                    if num_gpu > 1:
                        sample = model.module.sample()[0]
                    else:
                        sample = model.sample()[0]
                    model.train()

                grid_image = make_grid(
                    sample[0].cpu().detach().clamp(0, 1),
                    args.log.num_img_per_row, normalize=False, pad_value=1)
                writer.add_image('generation/1-image', grid_image, global_step)

                if args.arch.phase_background:
                    grid_image = make_grid(
                        sample[1].cpu().detach().clamp(0, 1),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('generation/2-fg', grid_image, global_step)

                    grid_image = make_grid(
                        sample[2].cpu().detach().clamp(0, 1),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('generation/3-alpha', grid_image, global_step)

                    grid_image = make_grid(
                        sample[3].cpu().detach().clamp(0, 1),
                        args.log.num_img_per_row, normalize=False, pad_value=1)
                    writer.add_image('generation/4-bg', grid_image, global_step)

                ###################################### generation end ######################################

                last_count = local_count

        ###################################### ll computing ######################################
        # only for logging, final ll should be computed using 100 particles

        if epoch % args.log.compute_nll_freq == 0:

            print(f'val nll at the end of epoch {epoch}')

            model.eval()
            args.log.phase_nll = True

            elbo_list = []
            kl_list = []
            ll_list = []
            with torch.no_grad():
                args.log.phase_log = False
                for batch_idx, sample in enumerate(val_loader):
                    imgs = sample.to(device)

                    ll_sample_list = []
                    for i in range(args.log.nll_num_sample):
                        _, log_like, kl, log_imp, _, _, _ = \
                            model(imgs)
                        aux_kl_pres, aux_kl_where, aux_kl_depth, aux_kl_what, \
                        aux_kl_bg, kl_pres, kl_where, kl_depth, kl_what, \
                        kl_global_all, kl_bg = kl

                        log_imp_pres, log_imp_depth, log_imp_what, log_imp_where, log_imp_bg, log_imp_g = log_imp

                        ll_sample_list.append(
                            (log_like + log_imp_pres + log_imp_depth + log_imp_what +
                             log_imp_where + log_imp_bg + log_imp_g).cpu()
                        )
                        # Only use one sample for elbo
                        if i == 0:
                            elbo_list.append((log_like - kl_pres - kl_where - kl_depth -
                                              kl_what - kl_bg - kl_global_all.sum(dim=1)).cpu())
                            kl_list.append((kl_pres + kl_where + kl_depth +
                                            kl_what + kl_bg + kl_global_all.sum(dim=1)).cpu())
                    ll_sample = log_mean_exp(torch.stack(ll_sample_list, dim=1), dim=1)
                    ll_list.append(ll_sample)

                ll_all = torch.cat(ll_list, dim=0)
                elbo_all = torch.cat(elbo_list, dim=0)
                kl_all = torch.cat(kl_list, dim=0)

            writer.add_scalar('val/ll', ll_all.mean(0).item(), global_step=epoch)
            writer.add_scalar('val/elbo', elbo_all.mean(0).item(), global_step=epoch)
            writer.add_scalar('val/kl', kl_all.mean(0).item(), global_step=epoch)

            args.log.phase_nll = False
            model.train()

        if epoch % (args.log.compute_nll_freq * 10) == 0:

            print(f'test nll at the end of epoch {epoch}')

            model.eval()
            args.log.phase_nll = True

            elbo_list = []
            kl_list = []
            ll_list = []
            with torch.no_grad():
                args.log.phase_log = False
                for batch_idx, sample in enumerate(test_loader):
                    imgs = sample.to(device)

                    ll_sample_list = []
                    for i in range(args.log.nll_num_sample):
                        _, log_like, kl, log_imp, _, _, _ = \
                            model(imgs)
                        aux_kl_pres, aux_kl_where, aux_kl_depth, aux_kl_what, \
                        aux_kl_bg, kl_pres, kl_where, kl_depth, kl_what, \
                        kl_global_all, kl_bg = kl

                        log_imp_pres, log_imp_depth, log_imp_what, log_imp_where, log_imp_bg, log_imp_g = log_imp

                        ll_sample_list.append(
                            (log_like + log_imp_pres + log_imp_depth + log_imp_what +
                             log_imp_where + log_imp_bg + log_imp_g).cpu()
                        )
                        # Only use one sample for elbo
                        if i == 0:
                            elbo_list.append((log_like - kl_pres - kl_where - kl_depth -
                                              kl_what - kl_bg - kl_global_all.sum(dim=1)).cpu())
                            kl_list.append((kl_pres + kl_where + kl_depth +
                                            kl_what + kl_bg + kl_global_all.sum(dim=1)).cpu())
                    ll_sample = log_mean_exp(torch.stack(ll_sample_list, dim=1), dim=1)
                    ll_list.append(ll_sample)

                ll_all = torch.cat(ll_list, dim=0)
                elbo_all = torch.cat(elbo_list, dim=0)
                kl_all = torch.cat(kl_list, dim=0)

            writer.add_scalar('test/ll', ll_all.mean(0).item(), global_step=epoch)
            writer.add_scalar('test/elbo', elbo_all.mean(0).item(), global_step=epoch)
            writer.add_scalar('test/kl', kl_all.mean(0).item(), global_step=epoch)

            args.log.phase_nll = False
            model.train()

        if epoch % args.log.save_epoch_freq == 0 and epoch != 0:
            save_ckpt(model_dir, model, optimizer, global_step, epoch,
                      local_count, args.train.batch_size, num_train)

    save_ckpt(model_dir, model, optimizer, global_step, epoch,
              local_count, args.train.batch_size, num_train)


def hyperparam_anneal(args, global_step):
    if args.train.beta_aux_pres_anneal_end_step == 0:
        args.train.beta_aux_pres = args.train.beta_aux_pres_anneal_start_value
    else:
        args.train.beta_aux_pres = linear_schedule(
            global_step,
            args.train.beta_aux_pres_anneal_start_step,
            args.train.beta_aux_pres_anneal_end_step,
            args.train.beta_aux_pres_anneal_start_value,
            args.train.beta_aux_pres_anneal_end_value
        )

    if args.train.beta_aux_where_anneal_end_step == 0:
        args.train.beta_aux_where = args.train.beta_aux_where_anneal_start_value
    else:
        args.train.beta_aux_where = linear_schedule(
            global_step,
            args.train.beta_aux_where_anneal_start_step,
            args.train.beta_aux_where_anneal_end_step,
            args.train.beta_aux_where_anneal_start_value,
            args.train.beta_aux_where_anneal_end_value
        )

    if args.train.beta_aux_what_anneal_end_step == 0:
        args.train.beta_aux_what = args.train.beta_aux_what_anneal_start_value
    else:
        args.train.beta_aux_what = linear_schedule(
            global_step,
            args.train.beta_aux_what_anneal_start_step,
            args.train.beta_aux_what_anneal_end_step,
            args.train.beta_aux_what_anneal_start_value,
            args.train.beta_aux_what_anneal_end_value
        )

    if args.train.beta_aux_depth_anneal_end_step == 0:
        args.train.beta_aux_depth = args.train.beta_aux_depth_anneal_start_value
    else:
        args.train.beta_aux_depth = linear_schedule(
            global_step,
            args.train.beta_aux_depth_anneal_start_step,
            args.train.beta_aux_depth_anneal_end_step,
            args.train.beta_aux_depth_anneal_start_value,
            args.train.beta_aux_depth_anneal_end_value
        )

    if args.train.beta_aux_global_anneal_end_step == 0:
        args.train.beta_aux_global = args.train.beta_aux_global_anneal_start_value
    else:
        args.train.beta_aux_global = linear_schedule(
            global_step,
            args.train.beta_aux_global_anneal_start_step,
            args.train.beta_aux_global_anneal_end_step,
            args.train.beta_aux_global_anneal_start_value,
            args.train.beta_aux_global_anneal_end_value
        )

    if args.train.beta_aux_bg_anneal_end_step == 0:
        args.train.beta_aux_bg = args.train.beta_aux_bg_anneal_start_value
    else:
        args.train.beta_aux_bg = linear_schedule(
            global_step,
            args.train.beta_aux_bg_anneal_start_step,
            args.train.beta_aux_bg_anneal_end_step,
            args.train.beta_aux_bg_anneal_start_value,
            args.train.beta_aux_bg_anneal_end_value
        )

    ########################### split here ###########################
    if args.train.beta_pres_anneal_end_step == 0:
        args.train.beta_pres = args.train.beta_pres_anneal_start_value
    else:
        args.train.beta_pres = linear_schedule(
            global_step,
            args.train.beta_pres_anneal_start_step,
            args.train.beta_pres_anneal_end_step,
            args.train.beta_pres_anneal_start_value,
            args.train.beta_pres_anneal_end_value
        )

    if args.train.beta_where_anneal_end_step == 0:
        args.train.beta_where = args.train.beta_where_anneal_start_value
    else:
        args.train.beta_where = linear_schedule(
            global_step,
            args.train.beta_where_anneal_start_step,
            args.train.beta_where_anneal_end_step,
            args.train.beta_where_anneal_start_value,
            args.train.beta_where_anneal_end_value
        )

    if args.train.beta_what_anneal_end_step == 0:
        args.train.beta_what = args.train.beta_what_anneal_start_value
    else:
        args.train.beta_what = linear_schedule(
            global_step,
            args.train.beta_what_anneal_start_step,
            args.train.beta_what_anneal_end_step,
            args.train.beta_what_anneal_start_value,
            args.train.beta_what_anneal_end_value
        )

    if args.train.beta_depth_anneal_end_step == 0:
        args.train.beta_depth = args.train.beta_depth_anneal_start_value
    else:
        args.train.beta_depth = linear_schedule(
            global_step,
            args.train.beta_depth_anneal_start_step,
            args.train.beta_depth_anneal_end_step,
            args.train.beta_depth_anneal_start_value,
            args.train.beta_depth_anneal_end_value
        )

    if args.train.beta_global_anneal_end_step == 0:
        args.train.beta_global = args.train.beta_global_anneal_start_value
    else:
        args.train.beta_global = linear_schedule(
            global_step,
            args.train.beta_global_anneal_start_step,
            args.train.beta_global_anneal_end_step,
            args.train.beta_global_anneal_start_value,
            args.train.beta_global_anneal_end_value
        )

    if args.train.tau_pres_anneal_end_step == 0:
        args.train.tau_pres = args.train.tau_pres_anneal_start_value
    else:
        args.train.tau_pres = linear_schedule(
            global_step,
            args.train.tau_pres_anneal_start_step,
            args.train.tau_pres_anneal_end_step,
            args.train.tau_pres_anneal_start_value,
            args.train.tau_pres_anneal_end_value
        )

    if args.train.beta_bg_anneal_end_step == 0:
        args.train.beta_bg = args.train.beta_bg_anneal_start_value
    else:
        args.train.beta_bg = linear_schedule(
            global_step,
            args.train.beta_bg_anneal_start_step,
            args.train.beta_bg_anneal_end_step,
            args.train.beta_bg_anneal_start_value,
            args.train.beta_bg_anneal_end_value
        )


    return


if __name__ == '__main__':
    main()
