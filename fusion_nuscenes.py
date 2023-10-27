import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
from PIL import Image

from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
from encoding.models.sseg import BaseNet
import torchvision.transforms as transforms

from fusion_util import extract_lseg_img_feature, PointCloudToImageMapper


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of LSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val"')
    parser.add_argument('--lseg_model', type=str, default='checkpoints/demo_e200.ckpt', help='Where is the LSeg checkpoint')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    
    # short hand
    split = args.split
    
    data_root_2d = args.data_root_2d
    point2img_mapper = args.point2img_mapper    
    evaluator = args.evaluator
    transform = args.transform
    cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

    # load 3D data (point cloud, color and the corresponding labels)
    # Only process points with GT label annotation
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2]
    mask_entire = labels_in!=255

    locs_in = locs_in[mask_entire]
    n_points = locs_in.shape[0]

    
    scene_id = data_path.split('/')[-1].split('.')[0]
    if exists(join(out_dir, scene_id +'.pt')):
        print(scene_id +'.pt' + ' already done!')
        return 1
    
    # process 2D features
    scene = join(data_root_2d, split, scene_id)
    img_dir_base = join(scene, 'color')
    pose_dir_base = join(scene, 'pose')
    K_dir_base = join(scene, 'K')
    num_img = len(cam_locs)
     
    device = torch.device('cpu')

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)


    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, cam in enumerate(tqdm(cam_locs)):
        intr = np.load(join(K_dir_base, cam+'.npy'))
        pose = np.load(join(pose_dir_base, cam+'.npy'))
        img_dir = join(img_dir_base, cam+'.jpg')

        # calculate the 3d-2d mapping
        mapping = np.ones([n_points_cur, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth=None, intrinsic=intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        feat_2d = extract_lseg_img_feature(img_dir, transform, evaluator).to(device)

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
        
        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]
    
    print(join(out_dir, scene_id +'.pt') + ' is saved!')

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (800, 450)
    ######################################
    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.feat_dim = 512
    split = args.split
    data_dir = args.data_dir
    args.img_dim = img_dim

    data_root = join(data_dir, 'nuscenes_3d')
    data_root_2d = join(data_dir,'nuscenes_2d')

    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    ##############################
    ##### load the LSeg model ####

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.lseg_model,
        data_path='../datasets/',
        dataset='ade20k',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )


    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module

    model = model.eval()
    model = model.cpu()

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    # THE trick for getting proper LSeg feature for Matterport
    scales = ([1])
    model.crop_size = 1600 # just use the long side!
    model.base_size = 1600

    evaluator = LSeg_MultiEvalModule(
        model, scales=scales, flip=True
    ).cuda() # LSeg model has to be in GPU
    evaluator.eval()

    args.evaluator = evaluator

    args.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()

    print("Arguments:")
    print(args)
    main(args)