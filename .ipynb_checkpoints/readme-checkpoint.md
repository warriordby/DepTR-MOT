git clone -b feature_align --single-branch https://gitee.com/warrior-deng/trackwithdepth.git

conda create env -f environment.yml
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt   ./SAM2/checkpoints/sam2.1_hiera_tiny.pt
wget https://hf-mirror.com/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth    ./VideoDepthAnything/video_depth_anything_vits.pth
cd SAM2
pip install -e 
cd ..



training:
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=4444 --nproc_per_node=2 train.py -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml --use-amp --seed=0


testing:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml -r ./output/dfine_hgnetv2_l_custom/epoch0.pth

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml --test-only -r ./output/dfine_hgnetv2_l_custom/last.pth


visual
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml --test-only -r ./output/dfine_hgnetv2_l_custom/last.pth -v


debug
.vscode/settings.json
按F5