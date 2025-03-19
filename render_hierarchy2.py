import math
import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
import torchvision
from random import randint
from argparse import ArgumentParser
from utils.loss_utils import ssim
from gaussian_renderer import render_post
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import sys
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

@torch.no_grad()
def render_set(args, scene, pipe, out_dir, tau, eval):
    render_path = out_dir

    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

    psnr_test, ssims, lpipss = 0.0, 0.0, 0.0
    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()

    # 使用 HDF5 存储中间结果
    h5_file_path = os.path.join(render_path, "intermediate_gaussians.h5")
    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)  # 添加这一行
    with h5py.File(h5_file_path, "w") as f:
        dset_xyz = f.create_dataset("xyz", shape=(0, 3), maxshape=(None, 3), dtype='float32', chunks=True)
        dset_opacities = f.create_dataset("opacities", shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
        dset_scales = f.create_dataset("scales", shape=(0, 3), maxshape=(None, 3), dtype='float32', chunks=True)
        dset_rotations = f.create_dataset("rotations", shape=(0, 4), maxshape=(None, 4), dtype='float32', chunks=True)

        for viewpoint in tqdm(cameras):
            viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
            viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
            viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
            viewpoint.camera_center = viewpoint.camera_center.cuda()

            threshold = 0
            to_render = expand_to_size(
                scene.gaussians.nodes, scene.gaussians.boxes, threshold,
                viewpoint.camera_center, torch.zeros((3)),
                render_indices, parent_indices, nodes_for_render_indices
            )
            
            indices = render_indices[:to_render].int().contiguous()
            node_indices = nodes_for_render_indices[:to_render].contiguous()

            get_interpolation_weights(
                node_indices, threshold, scene.gaussians.nodes, scene.gaussians.boxes,
                viewpoint.camera_center.cpu(), torch.zeros((3)),
                interpolation_weights, num_siblings
            )

            # 读取数据并转换为 numpy
            xyz_arr = scene.gaussians._xyz[indices].cpu().numpy()
            opacities_arr = scene.gaussians._opacity[indices].cpu().numpy().reshape(-1)
            scales_arr = scene.gaussians._scaling[indices].cpu().numpy().reshape(-1, 3)  # 确保是 (N, 3)
            rotations_arr = scene.gaussians._rotation[indices].cpu().numpy()

            # 追加写入 HDF5
            for dset, arr in [(dset_xyz, xyz_arr), (dset_opacities, opacities_arr),
                              (dset_scales, scales_arr)]:
                old_size = dset.shape[0]
                new_size = old_size + arr.shape[0]
                dset.resize((new_size,) + dset.shape[1:])  # 保持原有维度
                dset[old_size:new_size, ...] = arr

            # 追加写入 dset_rotations
            old_size = dset_rotations.shape[0]
            new_size = old_size + rotations_arr.shape[0]
            dset_rotations.resize((new_size, 4))
            dset_rotations[old_size:new_size, :] = rotations_arr

            # 每个循环结束后释放 GPU 缓存
            torch.cuda.empty_cache()

    # 读取 HDF5 数据并存回 GPU
    with h5py.File(h5_file_path, "r") as f:
        scene.gaussians._xyz = torch.tensor(f["xyz"][:]).cuda()
        scene.gaussians._opacity = torch.tensor(f["opacities"][:]).cuda()
        scene.gaussians._scaling = torch.tensor(f["scales"][:]).cuda()
        scene.gaussians._rotation = torch.tensor(f["rotations"][:]).cuda()

    # 保存合并的高斯模型
    save_ply_path = os.path.join(render_path, f"{tau}_concatenated_gaussians.ply")
    #scene.gaussians.save_ply(save_ply_path)

    scene.gaussians.save_ply_chunk(save_ply_path)
    
    if eval and len(scene.getTestCameras()) > 0:
        num_cameras = len(scene.getTestCameras())
        print(f"tau: {tau}, PSNR: {psnr_test/num_cameras:.5f}, SSIM: {ssims/num_cameras:.5f}, LPIPS: {lpipss/num_cameras:.5f}")


@torch.no_grad()
def render_set_export_ply_by_set(args, scene, pipe, out_dir, tau, eval):
    render_path = out_dir
    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()
    
    # scales=1.0
    # dataset_cameras = cameras[scales]
    total_samples = len(cameras)
    camera_indices=np.linspace(0, total_samples - 1, 30, dtype=int)
    camera_30 = [cameras[i] for i in camera_indices ] # 获取 CameraDataset
    
    
    
    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    
    psnr_test, ssims, lpipss = 0.0, 0.0, 0.0

    # 使用 HDF5 存储中间结果
    h5_file_path = os.path.join(render_path, "intermediate_gaussians.h5")
    # 检查文件是否存在
    if os.path.exists(h5_file_path):
        # 删除文件
        os.remove(h5_file_path)
        print(f"文件 {h5_file_path} 已删除")
    else:
        print(f"文件 {h5_file_path} 不存在")
    # Set 用于存储所有需要的高斯点索引
    all_indices = set()

    
    for viewpoint in tqdm(camera_30):
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()

        threshold = 0
        to_render = expand_to_size(
            scene.gaussians.nodes, scene.gaussians.boxes, threshold,
            viewpoint.camera_center, torch.zeros((3)),
            render_indices, parent_indices, nodes_for_render_indices
        )

        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices, threshold, scene.gaussians.nodes, scene.gaussians.boxes,
            viewpoint.camera_center.cpu(), torch.zeros((3)),
            interpolation_weights, num_siblings
        )

        # 将当前视角的索引加入集合
        all_indices.update(indices.cpu().numpy().ravel().tolist())

        # 每个循环结束后释放 GPU 缓存
        torch.cuda.empty_cache()
    
    
    # 将所有索引转换为唯一的 Tensor
    unique_indices = torch.tensor(list(all_indices), dtype=torch.long).cuda()


    if len(list(all_indices))==0:
        # 从 scene 中提取数据
        xyz_arr = scene.gaussians._xyz[unique_indices].cpu().numpy()
        opacities_arr = scene.gaussians._opacity[unique_indices].cpu().numpy().reshape(-1)
        scales_arr = scene.gaussians._scaling[unique_indices].cpu().numpy().reshape(-1, 3)
        rotations_arr = scene.gaussians._rotation[unique_indices].cpu().numpy()
    else:
        # test without extract by unique_indices
        xyz_arr = scene.gaussians._xyz.cpu().numpy()
        opacities_arr = scene.gaussians._opacity.cpu().numpy().reshape(-1)
        scales_arr = scene.gaussians._scaling.cpu().numpy().reshape(-1, 3)
        rotations_arr = scene.gaussians._rotation.cpu().numpy()


    # 使用 HDF5 存储
    with h5py.File(h5_file_path, "a") as f:
        # 如果文件中不存在数据集，则创建它们
        if 'xyz' not in f:
            dset_xyz = f.create_dataset('xyz', shape=(0, 3), maxshape=(None, 3), dtype='float32', chunks=(1000, 3), compression='gzip')
        else:
            dset_xyz = f['xyz']
        
        if 'opacities' not in f:
            dset_opacities = f.create_dataset('opacities', shape=(0,), maxshape=(None,), dtype='float32', chunks=(1000,), compression='gzip')
        else:
            dset_opacities = f['opacities']
        
        if 'scales' not in f:
            dset_scales = f.create_dataset('scales', shape=(0, 3), maxshape=(None, 3), dtype='float32', chunks=(1000, 3), compression='gzip')
        else:
            dset_scales = f['scales']
        
        if 'rotations' not in f:
            dset_rotations = f.create_dataset('rotations', shape=(0, 4), maxshape=(None, 4), dtype='float32', chunks=(1000, 4), compression='gzip')
        else:
            dset_rotations = f['rotations']

        # 追加写入数据到 HDF5
        for dset, arr in [(dset_xyz, xyz_arr), (dset_opacities, opacities_arr),
                          (dset_scales, scales_arr), (dset_rotations, rotations_arr)]:
            old_size = dset.shape[0]
            new_size = old_size + arr.shape[0]
            dset.resize((new_size,) + dset.shape[1:])  # 调整数据集大小
            dset[old_size:new_size, ...] = arr

    # 读取 HDF5 数据并存回 GPU
    with h5py.File(h5_file_path, "r") as f:
        scene.gaussians._xyz = torch.tensor(f["xyz"][:]).cuda()
        scene.gaussians._opacity = torch.tensor(f["opacities"][:]).cuda()
        scene.gaussians._scaling = torch.tensor(f["scales"][:]).cuda()
        scene.gaussians._rotation = torch.tensor(f["rotations"][:]).cuda()

    # 保存合并的高斯模型
    save_ply_path = os.path.join(render_path, f"{tau}_concatenated_gaussians.ply")
    
    #with indices
    #scene.gaussians.save_ply(save_ply_path,indices=unique_indices.cpu().numpy())
    #without indices
    #scene.gaussians.save_ply(save_ply_path,indices=None)
    
    #chunk_save
    if len(list(all_indices))==0:
        scene.gaussians.save_ply_chunk(save_ply_path)
    else:
        scene.gaussians.save_ply_chunk(save_ply_path, chunk_size=5_000_000,indices=unique_indices)

    if eval and len(scene.getTestCameras()) > 0:
        num_cameras = len(scene.getTestCameras())
        print(f"tau: {tau}, PSNR: {psnr_test/num_cameras:.5f}, SSIM: {ssims/num_cameras:.5f}, LPIPS: {lpipss/num_cameras:.5f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0])
    args = parser.parse_args(sys.argv[1:])

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)

    tau = 0.0  # 可遍历 args.taus
    #render_set(args, scene, pipe, os.path.join(args.out_dir, f"render_{tau}"), tau, args.eval)
    render_set_export_ply_by_set(args, scene, pipe, os.path.join(args.out_dir, f"render_{tau}"), tau, args.eval)
    