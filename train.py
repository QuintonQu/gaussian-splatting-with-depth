#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
import cv2 
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    Z_depths = []
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log_rgb = 0.0
    ema_loss_for_log_z = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    Z_total = 0.0
    l1_loss_total = 0.0

    extrapath = './experiments/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)

    extrapath = './experiments/gt'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
        
    extrapath = './experiments/res'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
        
    extrapath = './experiments/input'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        is_sonar = viewpoint_cam.is_sonar
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Loss
        gt_image = viewpoint_cam.original_image
        gt_density_h = viewpoint_cam.z_density_h if viewpoint_cam.z_density_h is not None else None
        gt_density_w = viewpoint_cam.z_density_w if viewpoint_cam.z_density_w is not None else None
        
        height = gt_image.shape[1]
        width = gt_image.shape[2]
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] 

        # print("image shape: ", image.shape)
        # print(torch.count_nonzero(image, dim=None))
        # print(render_pkg["z_density_h"].shape)
        # print(render_pkg["z_density_w"].shape)
        # print(gt_density_w.shape)
        # count the non-zero elements in the rendered density
        # print(torch.count_nonzero(render_pkg["z_density_h"], dim=None))
        # print(torch.count_nonzero(render_pkg["z_density_w"], dim=None))
        
        h_res_window = height // dataset.h_res
        w_res_window = width // dataset.w_res

        if not is_sonar:
            z_density_h = render_pkg["z_density_h"].unfold(1, h_res_window, h_res_window).sum(dim=2)
            z_density_w = render_pkg["z_density_w"].unfold(1, w_res_window, w_res_window).sum(dim=2)
        else: 
            z_density_h = render_pkg["z_density_h"]
            z_density_w = render_pkg["z_density_w"]

        # z_density_h = render_pkg["z_density_h"].unfold(1, h_res_window, h_res_window).sum(dim=2)
        # z_density_w = render_pkg["z_density_w"].unfold(1, w_res_window, w_res_window).sum(dim=2)
                
        # print("z_density_h shape: ", z_density_h.shape)
        # print("z_density_w shape: ", z_density_w.shape)

        # Min-max depth normalization
            
        #z_density_h = z_density_h / (z_density_h.max(dim=0, keepdim=True)[0] + 1e-10)
        z_density_w = z_density_w / (z_density_w.max(dim=0, keepdim=True)[0] + 1e-10)
        
        assert not torch.isnan(z_density_w).any(), "z_density_w is nan"

        if not is_sonar:
            # Q: is this necessary for sonar?
            assert z_density_h.shape[1] == dataset.h_res
            assert z_density_w.shape[1] == dataset.w_res

        if is_sonar:
            ZL = l2_loss(z_density_w, gt_density_w)
            # save the ground truth and rendered density as gray scale images
            
            #if torch.count_nonzero(z_density_w, dim=None) > 0:
            cv2.imwrite('./experiments/gt/gt_density_w_' + str(iteration) + '.png', gt_density_w.detach().cpu().numpy()*255)
            cv2.imwrite('./experiments/res/res_density_w_' + str(iteration) + '.png', z_density_w.detach().cpu().numpy()*255)

            # count the non-zero elements in the ground truth density
            #print("z_density_w ", torch.count_nonzero(z_density_w, dim=None))
            #print("g_denSity_w ", torch.count_nonzero(gt_density_w, dim=None)) 
            loss = ZL #/ (1.5 ** (iteration // opt.opacity_reset_interval + 1))
            Z_total += ZL.item()
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
            # if gt_density_h is not None and gt_density_w is not None:
            #     ZL = (l1_loss(z_density_h, gt_density_h) * dataset.w_res + l1_loss(z_density_w, gt_density_w) * dataset.h_res) / (dataset.h_res + dataset.w_res)
            #     if opt.depth_loss:
            #         loss += 0.1 * ZL / (1.5 ** (iteration // opt.opacity_reset_interval + 1))
            l1_loss_total += Ll1.item()

        #print("WARNING: backward is not called")
        loss.backward()
        iter_end.record()

        if iteration % 1000 == 0:
            print("ZL: ", Z_total / 1000)
            Z_depths.append((iteration, Z_total / 1000))
            Z_total = 0.0
            print("L1: ", l1_loss_total / 1000)
            l1_loss_total = 0.0

        with torch.no_grad():
            # Progress bar
            if not is_sonar:
                #ema_loss_for_log_rgb = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log_rgb
                #ema_loss_for_log_z = 0.4 * ZL.item() + 0.6 * ema_loss_for_log_z
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"L1": f"{Ll1.item():.{7}f}", "SSIM": f"{ssim(image, gt_image).item():.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                #training_report(tb_writer, iteration, Ll1, ZL, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            else: 
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Z_Loss": f"{ZL.item():.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                #Ll1 = torch.tensor(0.0)
                #training_report(tb_writer, iteration, Ll1, ZL, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # Log and save
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        # plot Z_depths 
    iterations = [x[0] for x in Z_depths]
    Z_depth = [x[1] for x in Z_depths]
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(iterations, Z_depth)
    plt.xlabel('Iterations')
    plt.ylabel('Z_depth')
    plt.title('Z_depth vs Iterations')
    plt.savefig('./experiments/Z_depth.png')



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, ZL, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/rgb_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/z_loss', ZL.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # WARNING: Modifying parameters here
    test_iterations = range(5000, args.iterations + 1, 3000)
    args.test_iterations = [i for i in test_iterations if i < args.iterations]
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
