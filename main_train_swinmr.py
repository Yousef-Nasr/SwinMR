"""
# -----------------------------------------
Main Program for Training
SwinMR for MRI_Recon
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
"""

import os
import sys
import math
import argparse
import random
import cv2
import numpy as np
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from data.dataset_CCsagnpi import safe_collate_fn, validate_data_files
from tensorboardX import SummaryWriter
from collections import OrderedDict
from skimage.transform import resize
import lpips
import platform
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[INFO] PIL not available - merged images will be saved without labels")


def get_optimal_workers(requested_workers, distributed=False, world_size=1):
    """
    Determine optimal number of workers based on platform and configuration.
    """
    import multiprocessing
    
    # Get system info
    max_cpu_workers = multiprocessing.cpu_count()
    platform_name = platform.system()
    
    if distributed:
        # For distributed training, divide workers among processes
        base_workers = min(requested_workers // max(world_size, 1), 4)
    else:
        base_workers = max(0, min(requested_workers, 4))
    
    # Platform-specific adjustments
    if platform_name == "Windows":
        # Windows has issues with multiprocessing in PyTorch
        optimal_workers = min(base_workers, 2)
        if optimal_workers > 0:
            print(f"[INFO] Windows detected - using {optimal_workers} workers (max recommended: 2)")
            print(f"[INFO] If you encounter multiprocessing errors, set dataloader_num_workers to 0")
    elif platform_name == "Darwin":  # macOS
        # macOS generally handles multiprocessing well but be conservative
        optimal_workers = min(base_workers, max_cpu_workers // 2)
    else:  # Linux and others
        optimal_workers = min(base_workers, max_cpu_workers)
    
    # Ensure we don't exceed system capabilities
    optimal_workers = max(0, min(optimal_workers, max_cpu_workers))
    
    return optimal_workers


def main(json_path=""):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    # opt['dist'] = parser.parse_args().dist

    # distributed settings
    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    # update opt
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"], net_type="G"
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"], net_type="E"
    )
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # save opt to  a '../option.json' file
    if opt["rank"] == 0:
        option.save(opt)

    # return None for missing key
    opt = option.dict_to_nonedict(opt)

    # configure logger
    if opt["rank"] == 0:
        # logger
        logger_name = "train"
        utils_logger.logger_info(
            logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
        )
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

        # tensorbordX log
        logger_tensorboard = SummaryWriter(os.path.join(opt["path"]["log"]))

    # set seed
    seed = opt["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    """

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
            )
            if opt["rank"] == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set,
                    shuffle=dataset_opt["dataloader_shuffle"],
                    drop_last=True,
                    seed=seed,
                )
                # Safe configuration for distributed training
                world_size = opt.get("world_size", 1)
                num_workers = get_optimal_workers(
                    dataset_opt["dataloader_num_workers"], 
                    distributed=True, 
                    world_size=world_size
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"] // opt["num_gpu"],
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=True,
                    pin_memory=False,
                    sampler=train_sampler,
                    collate_fn=safe_collate_fn,
                    persistent_workers=num_workers > 0,
                )
            else:
                # Safe configuration for single GPU training
                num_workers = get_optimal_workers(
                    dataset_opt["dataloader_num_workers"], 
                    distributed=False
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    num_workers=num_workers,
                    drop_last=True,
                    pin_memory=False,
                    collate_fn=safe_collate_fn,
                    persistent_workers=num_workers > 0,
                )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=0,  # Use single-threaded for test to avoid issues
                drop_last=False,
                pin_memory=False,
                collate_fn=safe_collate_fn,
            )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """
    # define model
    model = define_Model(opt)
    model.init_train()
    # define LPIPS function
    loss_fn_alex = lpips.LPIPS(net="alex").to(model.device)
    # define early stopping
    if opt["train"]["is_early_stopping"]:
        early_stopping = utils_early_stopping.EarlyStopping(
            patience=opt["train"]["early_stopping_num"]
        )

    # record
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """
    
    # Enhanced pre-training validation and setup
    if opt["rank"] == 0:
        print("\n" + "=" * 80)
        print("TRAINING INITIALIZATION")
        print("=" * 80)
        
        # Display configuration summary
        print(f"Model Type       : {opt.get('model', 'Unknown')}")
        print(f"Task             : {opt.get('task', 'Unknown')}")
        print(f"Batch Size       : {opt['datasets']['train']['dataloader_batch_size']}")
        print(f"Workers          : {opt['datasets']['train']['dataloader_num_workers']}")
        print(f"Learning Rate    : {opt['train']['G_optimizer_lr']:.2e}")
        print(f"Total Samples    : {len(train_set):,d}")
        print(f"Iterations/Epoch : {train_size:,d}")
        if opt.get('dist', False):
            print(f"Distributed      : Yes (World Size: {opt.get('world_size', 1)})")
        else:
            print(f"Distributed      : No")
        print(f"Test Every       : {opt['train']['checkpoint_test']:,d} iterations")
        print(f"Save Every       : {opt['train']['checkpoint_save']:,d} iterations")
        
        # Show checkpoint resumption info
        if current_step > 0:
            print(f"Resuming from    : Iteration {current_step:,d} (continuing training)")
        else:
            print(f"Starting fresh   : New training from iteration 1")
        
        print("-" * 80)
        
        logger.info("Validating data samples before training...")
        try:
            # Validate dataset paths if available
            if hasattr(train_set, 'paths_H') and len(train_set.paths_H) > 0:
                validate_data_files(train_set.paths_H, max_check=5)
                print("✓ Dataset file validation passed")
            
            # Test loading first batch
            print("✓ Testing first batch loading...")
            test_batch = next(iter(train_loader))
            if test_batch is not None:
                print("✓ First batch loaded successfully")
                logger.info("First batch loaded successfully")
                
                # Check tensor shapes and types
                if 'L' in test_batch and test_batch['L'] is not None:
                    print(f"  - L tensor shape: {test_batch['L'].shape}, dtype: {test_batch['L'].dtype}")
                    if torch.any(torch.isnan(test_batch['L'])) or torch.any(torch.isinf(test_batch['L'])):
                        logger.warning("⚠ NaN/Inf detected in training data L")
                        print("⚠ Warning: NaN/Inf detected in L data")
                    
                if 'H' in test_batch and test_batch['H'] is not None:
                    print(f"  - H tensor shape: {test_batch['H'].shape}, dtype: {test_batch['H'].dtype}")
                    if torch.any(torch.isnan(test_batch['H'])) or torch.any(torch.isinf(test_batch['H'])):
                        logger.warning("⚠ NaN/Inf detected in training data H")
                        print("⚠ Warning: NaN/Inf detected in H data")
                        
                print("✓ Data validation completed")
            else:
                logger.warning("First batch is None - potential data loading issues")
                print("⚠ Warning: First batch is None")
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            print(f"⚠ Data validation failed: {e}")
            logger.warning("Continuing with training, but expect potential issues...")
            print("Continuing with training...")
        
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print("Training in progress... (you should see iteration updates below)")
        
        # Suppress PyTorch scheduler warnings (known issue with MultiStepLR)
        import warnings
        warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step.*")
        warnings.filterwarnings("ignore", message="The epoch parameter in.*scheduler.step.*")
        
        print()

    for epoch in range(100000000):  # keep running
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        try:
            for i, train_data in enumerate(train_loader):
                current_step += 1

                # Show "Training started" message on first iteration
                if current_step == 1 and opt["rank"] == 0:
                    print("🚀 Training started! Iteration updates will appear below:")
                    print()

                # Skip batch if it's None (from safe_collate_fn)
                if train_data is None:
                    if opt["rank"] == 0:
                        print(f"\n⚠ Skipping batch {i} due to data loading issues")
                    logger.warning(f"Skipping batch {i} due to data loading issues")
                    continue

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                try:
                    model.feed_data(train_data)
                except Exception as e:
                    logger.error(f"Error feeding data at step {current_step}: {e}")
                    continue

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                try:
                    model.optimize_parameters(current_step)
                except Exception as e:
                    logger.error(f"Error optimizing parameters at step {current_step}: {e}")
                    continue

                # -------------------------------
                # 4) training information
                # -------------------------------
                if opt["rank"] == 0:
                    logs = model.current_log()
                    
                    # Show progress every iteration inline
                    progress_message = f"\r[{current_step:8,d}] Epoch:{epoch:3d} | LR:{model.current_learning_rate():.2e} | Loss:{logs.get('G_loss', 0):.4f}"
                    
                    # Add additional losses inline if available
                    if "G_loss_image" in logs:
                        progress_message += f" | Img:{logs['G_loss_image']:.4f}"
                    if "G_loss_frequency" in logs:
                        progress_message += f" | Freq:{logs['G_loss_frequency']:.4f}"
                    if "G_loss_preceptual" in logs:
                        progress_message += f" | Perc:{logs['G_loss_preceptual']:.4f}"
                    
                    # Print inline progress (overwrites previous line)
                    print(progress_message, end="", flush=True)
                    
                    # New line and detailed summary every 1000 iterations
                    if current_step % 1000 == 0:
                        print()  # New line after inline progress
                        print("=" * 80)
                        print(f"MILESTONE - Iteration {current_step:8,d} | Epoch {epoch:3d}")
                        print("=" * 80)
                        print(f"Learning Rate    : {model.current_learning_rate():.6e}")
                        print(f"Total Loss       : {logs.get('G_loss', 0):.6f}")
                        
                        # Calculate and show training speed
                        if hasattr(model, '_last_time'):
                            current_time = time.time()
                            time_per_1k_iters = current_time - model._last_time
                            iters_per_sec = 1000 / time_per_1k_iters if time_per_1k_iters > 0 else 0
                            print(f"Speed            : {iters_per_sec:.2f} iter/sec ({time_per_1k_iters:.1f}s/1k iters)")
                            model._last_time = current_time
                        else:
                            model._last_time = time.time()
                        
                        print("=" * 80)
                        print()  # Extra line for spacing
                
                # -------------------------------
                # 5) save model
                # -------------------------------
                if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                    print(f"\n💾 Saving model at iteration {current_step:,d}...")
                    logger.info("Saving the model.")
                    model.save(current_step)
                    print(f"✅ Model saved successfully!\n")
                
                # -------------------------------
                # 6) testing
                # -------------------------------
                if current_step % opt["train"]["checkpoint_test"] == 0 and opt["rank"] == 0:
                    print()
                    print("\n" + "=" * 80)
                    print(f"VALIDATION PHASE - Iteration {current_step:8,d}")
                    print("=" * 80)

                    # create folder for FID
                    img_dir_tmp_H = os.path.join(opt["path"]["images"], "tempH")
                    util.mkdir(img_dir_tmp_H)
                    img_dir_tmp_E = os.path.join(opt["path"]["images"], "tempE")
                    util.mkdir(img_dir_tmp_E)
                    img_dir_tmp_L = os.path.join(opt["path"]["images"], "tempL")
                    util.mkdir(img_dir_tmp_L)

                    # create result dict
                    test_results = OrderedDict()
                    test_results["psnr"] = []
                    test_results["ssim"] = []
                    test_results["lpips"] = []

                    test_results["G_loss"] = []
                    test_results["G_loss_image"] = []
                    test_results["G_loss_frequency"] = []
                    test_results["G_loss_preceptual"] = []

                    total_test_samples = len(test_loader)
                    print(f"Processing {total_test_samples} validation samples...")
                    print(f"Saving merged comparisons for first 20 samples...\n")
                    
                    for idx, test_data in enumerate(test_loader):
                        with torch.no_grad():
                            # Show detailed test progress for each sample
                            test_progress = f"\rTesting sample {idx + 1:3d}/{total_test_samples} ({((idx + 1) / total_test_samples) * 100:.1f}%) - Processing..."
                            print(test_progress, end="", flush=True)

                            img_info = test_data["img_info"][0]
                            img_dir = os.path.join(opt["path"]["images"], img_info)

                            # testing and adjust resolution
                            model.feed_data(test_data)
                            model.check_windowsize()
                            model.test()
                            model.recover_windowsize()

                            # acquire test result
                            results = model.current_results_gpu()

                            # calculate LPIPS (GPU | torch.tensor)
                            L_img = results["L"]
                            E_img = results["E"]
                            H_img = results["H"]
                            current_lpips = (
                                util.calculate_lpips_single(loss_fn_alex, H_img, E_img)
                                .data.squeeze()
                                .float()
                                .cpu()
                                .numpy()
                            )

                            # calculate PSNR SSIM (CPU | np.float)
                            L_img = util.tensor2float(L_img)
                            E_img = util.tensor2float(E_img)
                            H_img = util.tensor2float(H_img)
                            current_psnr = util.calculate_psnr_single(
                                H_img, E_img, border=0
                            )
                            current_ssim = util.calculate_ssim_single(
                                H_img, E_img, border=0
                            )

                            # record metrics
                            test_results["psnr"].append(current_psnr)
                            test_results["ssim"].append(current_ssim)
                            test_results["lpips"].append(current_lpips)

                            # Save individual samples (first 5 only)
                            if idx < 5:
                                util.mkdir(img_dir)
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir, "ZF_{:05d}.png".format(current_step)
                                    ),
                                    np.clip(L_img, 0, 1) * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir, "Recon_{:05d}.png".format(current_step)
                                    ),
                                    np.clip(E_img, 0, 1) * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir, "GT_{:05d}.png".format(current_step)
                                    ),
                                    np.clip(H_img, 0, 1) * 255,
                                )
                            
                            # Save merged comparison images (first 20 samples)
                            if idx < 20:
                                # Create merged image with labels: GT | Noisy | Predicted
                                h, w = H_img.shape[:2]
                                label_height = 30
                                merged_img = np.ones((h + label_height, w * 3), dtype=np.float32)
                                
                                # Place images side by side (offset by label height)
                                merged_img[label_height:, :w] = H_img  # Ground Truth
                                merged_img[label_height:, w:2*w] = L_img  # Noisy (Zero-filled)
                                merged_img[label_height:, 2*w:3*w] = E_img  # Predicted (Reconstructed)
                                
                                # Create merged results directory
                                merged_dir = os.path.join(opt["path"]["images"], "merged_comparisons")
                                util.mkdir(merged_dir)
                                
                                if PIL_AVAILABLE:
                                    # Convert to PIL for text labels
                                    merged_pil = Image.fromarray((np.clip(merged_img, 0, 1) * 255).astype(np.uint8))
                                    draw = ImageDraw.Draw(merged_pil)
                                    
                                    # Add labels (use default font)
                                    try:
                                        draw.text((w//2-30, 5), "Ground Truth", fill=0)
                                        draw.text((w+w//2-20, 5), "Noisy Input", fill=0)
                                        draw.text((2*w+w//2-25, 5), "Predicted", fill=0)
                                    except:
                                        pass  # Skip labels if font issues
                                    
                                    # Save merged image with labels
                                    merged_pil.save(
                                        os.path.join(
                                            merged_dir, f"comparison_{current_step:05d}_sample_{idx:03d}.png"
                                        )
                                    )
                                else:
                                    # Fallback: save without labels using cv2
                                    cv2.imwrite(
                                        os.path.join(
                                            merged_dir, f"comparison_{current_step:05d}_sample_{idx:03d}.png"
                                        ),
                                        np.clip(merged_img, 0, 1) * 255,
                                    )

                            # Save temp images for FID calculation
                            if opt["datasets"]["test"].get("resize_for_fid", False):
                                resize_for_fid = opt["datasets"]["test"]["resize_for_fid"]
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_L, "ZF_{:05d}.png".format(idx)
                                    ),
                                    resize(
                                        np.clip(L_img, 0, 1),
                                        (resize_for_fid[0], resize_for_fid[1]),
                                    )
                                    * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_E, "Recon_{:05d}.png".format(idx)
                                    ),
                                    resize(
                                        np.clip(E_img, 0, 1),
                                        (resize_for_fid[0], resize_for_fid[1]),
                                    )
                                    * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_H, "GT_{:05d}.png".format(idx)
                                    ),
                                    resize(
                                        np.clip(H_img, 0, 1),
                                        (resize_for_fid[0], resize_for_fid[1]),
                                    )
                                    * 255,
                                )
                            else:
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_L, "ZF_{:05d}.png".format(idx)
                                    ),
                                    np.clip(L_img, 0, 1) * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_E, "Recon_{:05d}.png".format(idx)
                                    ),
                                    np.clip(E_img, 0, 1) * 255,
                                )
                                cv2.imwrite(
                                    os.path.join(
                                        img_dir_tmp_H, "GT_{:05d}.png".format(idx)
                                    ),
                                    np.clip(H_img, 0, 1) * 255,
                                )

                    print("\n")  # New line after test progress completion
                    print("-" * 80)
                    print("VALIDATION RESULTS")
                    print("-" * 80)

                    # summarize psnr/ssim/lpips
                    ave_psnr = np.mean(test_results["psnr"])
                    ave_ssim = np.mean(test_results["ssim"])
                    ave_lpips = np.mean(test_results["lpips"])

                    # calculate FID
                    if opt["dist"]:
                        # DistributedDataParallel (If multiple GPUs are used to train, use the 2nd GPU for FID calculation.)
                        log = os.popen(
                            "{} -m pytorch_fid {} {} ".format(
                                sys.executable, img_dir_tmp_H, img_dir_tmp_E
                            )
                        ).read()
                    else:
                        # DataParallel (If multiple GPUs are used to train, use the 2nd GPU for FID calculation for unbalance of GPU menory use.)
                        if len(opt["gpu_ids"]) > 1:
                            log = os.popen(
                                "{} -m pytorch_fid --device cuda:1 {} {} ".format(
                                    sys.executable, img_dir_tmp_H, img_dir_tmp_E
                                )
                            ).read()
                        else:
                            log = os.popen(
                                "{} -m pytorch_fid {} {} ".format(
                                    sys.executable, img_dir_tmp_H, img_dir_tmp_E
                                )
                            ).read()
                    print(log)
                    fid = eval(log.replace("FID:  ", ""))

                    # Enhanced testing log with better formatting
                    print(f"Average PSNR     : {ave_psnr:8.4f} dB")
                    print(f"Average SSIM     : {ave_ssim:8.6f}")
                    print(f"Average LPIPS    : {ave_lpips:8.6f}")
                    print(f"FID Score        : {fid:8.4f}")
                    print("=" * 80)
                    print()
                    
                    logger.info(
                        "<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.4f}; Average SSIM : {:<.6f}; LPIPS : {:<.6f}; FID : {:<.4f}".format(
                            epoch, current_step, ave_psnr, ave_ssim, ave_lpips, fid
                        )
                    )

                    logger_tensorboard.add_scalar(
                        "VALIDATION PSNR", ave_psnr, global_step=current_step
                    )
                    logger_tensorboard.add_scalar(
                        "VALIDATION SSIM", ave_ssim, global_step=current_step
                    )
                    logger_tensorboard.add_scalar(
                        "VALIDATION LPIPS", ave_lpips, global_step=current_step
                    )
                    logger_tensorboard.add_scalar(
                        "VALIDATION FID", fid, global_step=current_step
                    )

            # Detailed logging at checkpoint intervals
            if (
                current_step % opt["train"]["checkpoint_print"] == 0
                and opt["rank"] == 0
            ):
                logs = model.current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.3e} ".format(k, v)
                logger.info(message)

                # record train loss
                logger_tensorboard.add_scalar(
                    "Learning Rate",
                    model.current_learning_rate(),
                    global_step=current_step,
                )
                logger_tensorboard.add_scalar(
                    "TRAIN Generator LOSS/G_loss",
                    logs["G_loss"],
                    global_step=current_step,
                )

                if "G_loss_image" in logs.keys():
                    logger_tensorboard.add_scalar(
                        "TRAIN Generator LOSS/G_loss_image",
                        logs["G_loss_image"],
                        global_step=current_step,
                    )
                if "G_loss_frequency" in logs.keys():
                    logger_tensorboard.add_scalar(
                        "TRAIN Generator LOSS/G_loss_frequency",
                        logs["G_loss_frequency"],
                        global_step=current_step,
                    )
                if "G_loss_preceptual" in logs.keys():
                    logger_tensorboard.add_scalar(
                        "TRAIN Generator LOSS/G_loss_preceptual",
                        logs["G_loss_preceptual"],
                        global_step=current_step,
                    )

                # (Moved to inside training loop above)
        
        except Exception as e:
            if opt["rank"] == 0:
                print(f"\n⚠ Training epoch {epoch} failed with error: {e}")
                logger.error(f"Training epoch {epoch} failed with error: {e}")
                logger.info("Continuing to next epoch...")
                print("Continuing to next epoch...\n")
            continue

    print("Training Stop")


if __name__ == "__main__":
    main()
