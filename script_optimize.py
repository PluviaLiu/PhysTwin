import glob
import os
import json
import time

script_t0 = time.perf_counter()

base_path = "./data/different_types"

scan_t0 = time.perf_counter()
dir_names = glob.glob(f"{base_path}/*")
print(
    f"[TIMER][script_optimize] 扫描 cases: {time.perf_counter() - scan_t0:.3f}s, "
    f"num_cases={len(dir_names)}"
)

for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    case_t0 = time.perf_counter()
    print(f"[TIMER][script_optimize][{case_name}] 开始")

    # Read the train test split
    read_t0 = time.perf_counter()
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]
    print(
        f"[TIMER][script_optimize][{case_name}] 读取 split.json: "
        f"{time.perf_counter() - read_t0:.3f}s, train_frame={train_frame}"
    )

    call_t0 = time.perf_counter()
    ret = os.system(
        f"python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )
    print(
        f"[TIMER][script_optimize][{case_name}] 调用 optimize_cma.py: "
        f"{time.perf_counter() - call_t0:.3f}s, return_code={ret}"
    )
    print(
        f"[TIMER][script_optimize][{case_name}] case total: "
        f"{time.perf_counter() - case_t0:.3f}s"
    )

print(f"[TIMER][script_optimize] total: {time.perf_counter() - script_t0:.3f}s")
