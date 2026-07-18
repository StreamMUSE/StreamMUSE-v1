"""统计每首歌的 onset 密度，筛除最低5%和最高5%"""

import os
import sys
import numpy as np
from multiprocessing import Pool
from functools import partial

DATA_DIR = "/DATA7_6T/cby/musicxml/allxml_npz_dual_track_optimized_no_underscore"
OUTPUT_DIR = "/DATA7_6T/cby/musicxml/allxml_npz_filtered"


def compute_onset_density(fname, data_dir):
    """计算单个文件的 onset 密度 = total_onsets / total_timesteps"""
    try:
        path = os.path.join(data_dir, fname)
        d = np.load(path, allow_pickle=True)
        meta = d['metadata'].item()
        num_measures = meta['num_measures']

        total_onsets = 0
        total_timesteps = 0
        for i in range(num_measures):
            m = d[f'measure_{i}']  # (4, 88, T)
            # channel 0: melody onset, channel 2: accompaniment onset
            total_onsets += np.count_nonzero(m[0]) + np.count_nonzero(m[2])
            total_timesteps += m.shape[2]

        density = total_onsets / total_timesteps if total_timesteps > 0 else 0
        return fname, density, num_measures
    except Exception as e:
        return fname, -1, 0


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
    print(f"共 {len(files)} 个文件，开始计算 onset 密度...")

    # 多进程加速
    worker = partial(compute_onset_density, data_dir=DATA_DIR)
    with Pool(16) as pool:
        results = pool.map(worker, files, chunksize=200)

    # 过滤掉出错的
    valid = [(f, d, n) for f, d, n in results if d >= 0]
    failed = [f for f, d, _ in results if d < 0]
    print(f"有效: {len(valid)}, 失败: {len(failed)}")

    densities = np.array([d for _, d, _ in valid])
    fnames = [f for f, _, _ in valid]

    # 统计
    print(f"\nOnset 密度分布:")
    print(f"  min={densities.min():.4f}, max={densities.max():.4f}")
    print(f"  mean={densities.mean():.4f}, median={np.median(densities):.4f}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d} = {np.percentile(densities, p):.4f}")

    # 筛选: 去掉最低5%和最高5%
    low = np.percentile(densities, 5)
    high = np.percentile(densities, 95)
    keep_mask = (densities >= low) & (densities <= high)
    keep_files = [fnames[i] for i in range(len(fnames)) if keep_mask[i]]
    remove_files = [fnames[i] for i in range(len(fnames)) if not keep_mask[i]]

    print(f"\n筛选阈值: [{low:.4f}, {high:.4f}]")
    print(f"保留: {len(keep_files)} ({len(keep_files)/len(valid)*100:.1f}%)")
    print(f"移除: {len(remove_files)} ({len(remove_files)/len(valid)*100:.1f}%)")

    # 创建符号链接目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in keep_files:
        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(f"\n已创建筛选后数据集 (符号链接): {OUTPUT_DIR}")
    print(f"共 {len(keep_files)} 个文件")


if __name__ == "__main__":
    main()
