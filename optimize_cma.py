# The first stage to optimize the sparse parameters using CMA-ES
from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import torch
import pickle
import json
import time
from argparse import ArgumentParser


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timer_msg(case_name, stage_name, start_time, extra=""):
    msg = (
        f"[TIMER][optimize_cma][{case_name}] {stage_name}: "
        f"{time.perf_counter() - start_time:.3f}s"
    )
    if extra:
        msg += f", {extra}"
    logger.info(msg)


seed_t0 = time.perf_counter()
seed = 42
set_all_seeds(seed)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    total_t0 = time.perf_counter()

    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter

    timer_msg(case_name, "seed/setup", seed_t0, extra=f"seed={seed}")

    config_t0 = time.perf_counter()
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
        config_name = "configs/cloth.yaml"
    else:
        cfg.load_from_yaml("configs/real.yaml")
        config_name = "configs/real.yaml"
    timer_msg(case_name, "load config", config_t0, extra=config_name)

    base_dir = f"experiments_optimization/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    calibrate_t0 = time.perf_counter()
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    timer_msg(case_name, "load calibrate.pkl", calibrate_t0, extra=f"num_cameras={len(c2ws)}")

    metadata_t0 = time.perf_counter()
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"
    timer_msg(case_name, "load metadata.json", metadata_t0, extra=f"WH={cfg.WH}")

    log_t0 = time.perf_counter()
    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    timer_msg(case_name, "set log file", log_t0, extra=base_dir)

    optimizer_t0 = time.perf_counter()
    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    timer_msg(case_name, "create OptimizerCMA", optimizer_t0)

    optimize_t0 = time.perf_counter()
    optimizer.optimize(max_iter=max_iter)
    timer_msg(case_name, "optimizer.optimize", optimize_t0, extra=f"max_iter={max_iter}")

    timer_msg(case_name, "total", total_t0)
