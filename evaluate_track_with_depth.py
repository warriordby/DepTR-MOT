import os
import os.path as osp
import time
import cv2
from loguru import logger
from ByteTrack.yolox.exp import get_exp
from ByteTrack.tools.demo_track_frome_detdtxt import Predictor, image_demo   # 假设你的代码保存成 your_demo_script.py
from DanceTrack.TrackEval.trackeval import Evaluator  # pip install trackeval
import time  # 用于计时

def evaluate_dancetrack(gt_root, result_root):
    """使用 TrackEval 计算指标"""
    from DanceTrack.TrackEval import trackeval
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 4,
        'PRINT_RESULTS': True,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
    }
    dataset_config = {
        'GT_FOLDER': gt_root,         # dancetrack/val
        'TRACKERS_FOLDER': result_root, # 保存的tracking结果
        'SEQMAP_FILE': None,
        'SEQ_INFO': None,
        'BENCHMARK': 'dancetrack',
        'SEQMAP_FOLDER': '../QuadTrack/seqmaps',
        'SPLIT_TO_EVAL': 'test'
    }
    metrics_config = {'METRICS': ['HOTA','CLEAR','Identity']}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate(dataset_list, metrics_list)

def run_all_sequences(val_root, det_root, exp_file, exp_name):
    exp = get_exp(exp_file, None)
    output_dir = osp.join(exp.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_root = output_dir
    os.makedirs(vis_root, exist_ok=True)
    fps_list = []
    
    # seqs = sorted(os.listdir(val_root))
    # for seq in seqs:
    #     img_dir = osp.join(val_root, seq, "img1")
    #     det_file = osp.join(det_root, f"{seq}.txt")
    #     if not osp.exists(img_dir) or not osp.exists(det_file):
    #         logger.warning(f"skip {seq}, no {img_dir} or det {det_file}")
    #         continue

    #     logger.info(f"Processing {seq}")
    #     predictor = Predictor(detpath=det_file, exp=exp)
    #     current_time = time.localtime()

    #     # 跑 image_demo
    #     from types import SimpleNamespace
    #     args = SimpleNamespace(
    #         demo="image",
    #         path=img_dir,
    #         save_result=True,
    #         fps=30,
    #         aspect_ratio_thresh=1.6,
    #         min_box_area=10,
    #         track_thresh=0.5,
    #         track_buffer=30,
    #         match_thresh=0.8,
    #         mot20=None,
    #     )
    #     total_fps   = image_demo(predictor, vis_root, seq, current_time, exp, args)
    #     fps_list.append(total_fps)
    #     # break
    # average_fps = sum(fps_list) / len(fps_list)
    # print(f"Average FPS across sequences: {average_fps:.2f}")
    # 全部跑完后，做评估
    evaluate_dancetrack(val_root, exp.output_dir)


if __name__ == "__main__":
    #
    run_all_sequences(
        val_root=r"/root/autodl-tmp/QuadTrack/test",
        det_root=r"./visual_out/detection_results",
        exp_file="./ByteTrack/exps/example/mot/yolox_x_mix_det.py",
        exp_name="DepthMOT"
    )
