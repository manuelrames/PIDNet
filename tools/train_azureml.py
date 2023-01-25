# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel

from azureml.core import Run, Model, Dataset, Workspace


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--training-dataset", type=str, dest='training_dataset', help='training dataset')
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=None, dest='batch_size', help='training batch size')
    parser.add_argument("--learning-rate", type=float, default=None, dest='learning_rate', help='training learning rate')
    parser.add_argument("--max-num-epochs", type=int, default=None, dest='max_num_epochs', help='maximum number of training epochs')
    parser.add_argument("--momentum", type=float, default=None, dest='momentum', help='training momentum')
    parser.add_argument("--weight-decay", type=float, default=None, dest='weight_decay', help='learning rate weight decay')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # Get the experiment run context
    run = Run.get_context()

    # Load the workspace
    ws = run.experiment.workspace

    # Load the data (passed as an input dataset)
    print("Loading Data...")
    training_data = run.input_datasets['railways_semsegm_5classes_dataset']
    print("Data Loaded!")

    # Download train/test data to workspace and show some examples
    print("Downloading train/test data to workspace")
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    print("Created data folder %s" % data_folder)
    training_data.download(target_path=data_folder)
    print("All Data Was Downloaded!")

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict, run)

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)

            # log validation loss and mean IoU to AureML
            run.log('valid_loss', valid_loss)
            run.log('mean_IoU', mean_IoU)

        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)



    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int_((end-start)/3600))
    logger.info('Done')

    # Upload tensorboard logdir
    run.upload_folder(name='log', path='log')

    # Datasets used
    training_ds = Dataset.get_by_name(ws, name='railways_semsegm_5classes_dataset')

    # Performance metrics on validation dataset
    #perf_metrics = {'best_mIoU': best_mIoU, 'mean_IoU': mean_IoU, 'valid_loss': valid_loss, 'background_IoU': IoU_array[0],
    #                'train_rail_IoU': IoU_array[1], 'ballast_IoU': IoU_array[2]}
    perf_metrics = {'best_mIoU': best_mIoU, 'mean_IoU': mean_IoU, 'valid_loss': valid_loss, 'background_IoU': IoU_array[0],
                    'left_rail_IoU': IoU_array[1], 'right_rail_IoU': IoU_array[2], 'other_rail_IoU': IoU_array[3], 'irrig_zone_IoU': IoU_array[4],
                    'no_irrig_zone_IoU': IoU_array[5]}

    # Register the model
    print('Registering best model...')
    model = run.register_model(model_name=f"railways_SemSegm_5classes_{config.MODEL.NAME}",
                               model_path=os.path.join(f"outputs/{config.DATASET.DATASET}/{config.MODEL.NAME}_{config.DATASET.DATASET}", "best.pt"),
                               # model_framework = Model.Framework.PYTORCH,
                               model_framework='Custom',
                               # 'PyTorch' seems not to be supported as of 13/07/2022... -> https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#azureml-core-run-run-register-model
                               datasets=[('railways_semsegm_5classes_dataset', training_ds)],
                               tags={'model_size': config.MODEL.NAME,
                                     'num_classes': config.DATASET.NUM_CLASSES,
                                     'loss_class_balance': str(config.LOSS.CLASS_BALANCE),
                                     'model_pretrained': config.MODEL.PRETRAINED,
                                     'batch_size': config.TRAIN.BATCH_SIZE_PER_GPU,
                                     'max_num_epochs': config.TRAIN.END_EPOCH,
                                     'learning_rate': config.TRAIN.LR,
                                     'momentum': config.TRAIN.MOMENTUM,
                                     'optimizer': config.TRAIN.OPTIMIZER,
                                     'weight_decay': config.TRAIN.WD},
                               description=f"PIDNet model trained for railway semantic segmentation. It will be used to calculate the irrigation zone as well as the distance vegetation-train.",
                               properties=perf_metrics)

    print('Name:', model.name)
    print('Version:', model.version)

    # End the run
    run.complete()

if __name__ == '__main__':
    main()
