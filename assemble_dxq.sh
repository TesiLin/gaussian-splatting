#!/bin/bash
set -e
echo "PATH: $PATH"
echo "Which conda: $(which conda)"
echo "Which python: $(which python)"

cd /workspace/code/gaussian-splatting-tutorial/code/3dgs
echo "cd dir"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate gs
echo "Conda environment 'gs' activated"

get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

run_mesh() {
  local depth_trunc="$1"
  local voxel_size="$2"
  local num_cluster="${3:-50}"

  local sdf_trunc
  sdf_trunc=$(awk -v v="$voxel_size" 'BEGIN { printf "%.6f", v * 4 }')

  local tag_depth tag_voxel tag_sdf
  tag_depth=$(echo "$depth_trunc" | sed 's/\./p/g')
  tag_voxel=$(echo "$voxel_size" | sed 's/\./p/g')
  tag_sdf=$(echo "$sdf_trunc" | sed 's/\./p/g')

  local mesh_tag="d${tag_depth}_v${tag_voxel}_s${tag_sdf}_c${num_cluster}"

  echo "Running mesh extraction:"
  echo "  depth_trunc = $depth_trunc"
  echo "  voxel_size  = $voxel_size"
  echo "  sdf_trunc   = $sdf_trunc"
  echo "  num_cluster = $num_cluster"
  echo "  mesh_tag    = $mesh_tag"

  python extract_mesh.py \
    -s "$DATASET" \
    -m "$MODEL" \
    --voxel_size "$voxel_size" \
    --sdf_trunc "$sdf_trunc" \
    --depth_trunc "$depth_trunc" \
    --num_cluster "$num_cluster" \
    --mesh_tag "$mesh_tag"
}

run_mesh_by_depth() {
  local depth_trunc="$1"
  local num_cluster="${2:-50}"

  local voxel_size
  voxel_size=$(awk -v d="$depth_trunc" 'BEGIN { printf "%.6f", d / 500 }')

  run_mesh "$depth_trunc" "$voxel_size" "$num_cluster"
}

DATASET="/workspace/3Ddataset/dxq0629_bbox_959_1961"
MODEL="output/dxq0629_bbox_959_1961"

python train.py -s "$DATASET" -m "$MODEL"

python render.py -s "$DATASET" -m "$MODEL"

run_mesh 3 0.04 5