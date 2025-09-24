# Video-Depth-Anything for Metric Depth Estimation
We here provide a simple demo for our fine-tuned Video-Depth-Anything metric model. We fine-tune our pre-trained model on Virtual KITTI and IRS datasets for metric depth estimation. 

# Pre-trained Models
| Base Model | Params | Checkpoint |
|:-|-:|:-:|
| Video-Depth-Anything-V2-Large-Metric | 381.8M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth) |

# Usage
## Preparation
Download the checkpoints and put them under the `metric_depth/checkpoints` directory.

## Use our models
### Running script on video
```bash
cd metric_depth
python3 run.py \
    --input_video <YOUR_VIDEO_PATH> \
    --output_dir <YOUR_OUTPUT_DIR>
```
### Project video to point clouds
```bash
cd metric_depth
python3 depth_to_pointcloud.py \
    --input_video <YOUR_VIDEO_PATH> \
    --output_dir <YOUR_OUTPUT_DIR> \
    --focal-length-x <CAMERA FX> \
    --focal-length-y <CAMERA FY> \
```
