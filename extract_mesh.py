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

import torch
import os
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh
import open3d as o3d


if __name__ == "__main__":
    parser = ArgumentParser(description="Mesh extraction parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--mesh_tag", default="", type=str, help="Mesh output tag")
    args = get_combined_args(parser)
    print("Extracting mesh from " + args.model_path)

    safe_state(args.quiet)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussExtractor = GaussianExtractor(gaussians, render, pipe=pipeline.extract(args), bg_color=bg_color)

    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        train_cameras = scene.getTrainCameras()
        if scene.use_lazy_loading:
            scene.load_camera_batch(train_cameras, list(range(len(train_cameras))))
        gaussExtractor.reconstruction(train_cameras)
        gaussExtractor.export_image(train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        test_cameras = scene.getTestCameras()
        if scene.use_lazy_loading:
            scene.load_camera_batch(test_cameras, list(range(len(test_cameras))))
        gaussExtractor.reconstruction(test_cameras)
        gaussExtractor.export_image(test_dir)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # Use training cameras for mesh extraction
        train_cameras = scene.getTrainCameras()
        if scene.use_lazy_loading:
            scene.load_camera_batch(train_cameras, list(range(len(train_cameras))))
        gaussExtractor.reconstruction(train_cameras)

        tag = f"_{args.mesh_tag}" if args.mesh_tag else ""

        if args.unbounded:
            name = f'fuse_unbounded{tag}.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)

            gaussExtractor.export_depth_stats(
                train_dir,
                filename=f"depth_stats_unbounded{tag}.json",
                extra={
                    "unbounded": True,
                    "mesh_res": args.mesh_res,
                    "num_cluster": args.num_cluster,
                    "mesh_tag": args.mesh_tag,
                }
            )
        else:
            name = f'fuse{tag}.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc

            gaussExtractor.export_depth_stats(
                train_dir,
                filename=f"depth_stats{tag}.json",
                extra={
                    "unbounded": False,
                    "voxel_size": voxel_size,
                    "sdf_trunc": sdf_trunc,
                    "depth_trunc": depth_trunc,
                    "num_cluster": args.num_cluster,
                    "mesh_res": args.mesh_res,
                    "mesh_tag": args.mesh_tag,
                    "estimated_radius": float(gaussExtractor.radius),
                }
            )

            mesh = gaussExtractor.extract_mesh_bounded(
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
            )

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
