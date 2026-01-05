import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def clean_single_file(npz_path, source_dir, target_dir):
    """
    处理单个文件的函数
    
    Returns:
        tuple: (文件名, 状态('kept'/'removed'), 错误信息)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        metadata = data['metadata'].item()
        
        # 收集非空的segments
        valid_segments = []
        valid_indices = []
        
        i = 0
        while f'measure_{i}' in data:
            segment = data[f'measure_{i}']
            if np.any(segment > 0):
                valid_segments.append(segment)
                valid_indices.append(i)
            i += 1
        
        # 如果没有有效的segment，返回删除状态
        if len(valid_segments) == 0:
            return Path(npz_path).name, 'removed', None
        
        # 更新metadata
        metadata['num_measures'] = len(valid_segments)
        metadata['original_measures'] = i
        metadata['valid_measures'] = valid_indices
        
        # 确定输出路径
        if target_dir and target_dir != source_dir:
            output_path = Path(target_dir) / Path(npz_path).name
        else:
            output_path = npz_path
        
        # 保存清理后的数据
        save_dict = {f'measure_{j}': seg.astype(np.uint8) 
                     for j, seg in enumerate(valid_segments)}
        save_dict['metadata'] = metadata
        
        np.savez_compressed(output_path, **save_dict)
        return Path(npz_path).name, 'kept', None
        
    except Exception as e:
        return Path(npz_path).name, 'error', str(e)


def batch_clean_dataset_parallel(source_dir: str, target_dir: str = None, max_workers: int = 8):
    """
    并行批量清理数据集
    
    Args:
        source_dir: 源数据目录
        target_dir: 目标目录（None则原地清理）
        max_workers: 最大进程数
    """
    if target_dir and target_dir != source_dir:
        os.makedirs(target_dir, exist_ok=True)
    
    npz_files = list(Path(source_dir).glob('*.npz'))
    print(f"Processing {len(npz_files)} files with {max_workers} workers...")
    
    removed_files = []
    cleaned_files = []
    error_files = []
    
    # 创建partial函数，固定source_dir和target_dir参数
    process_func = partial(clean_single_file, 
                          source_dir=source_dir, 
                          target_dir=target_dir)
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_func, str(npz_path)): npz_path 
                  for npz_path in npz_files}
        
        # 使用tqdm显示进度
        with tqdm(total=len(npz_files), desc="Cleaning") as pbar:
            for future in as_completed(futures):
                filename, status, error = future.result()
                
                if status == 'kept':
                    cleaned_files.append(filename)
                    pbar.set_description(f"✓ {filename[:30]}")
                elif status == 'removed':
                    removed_files.append(filename)
                    # 如果是原地清理，删除源文件
                    if target_dir is None or target_dir == source_dir:
                        os.remove(futures[future])
                    pbar.set_description(f"✗ {filename[:30]}")
                else:  # error
                    error_files.append((filename, error))
                    pbar.set_description(f"⚠ {filename[:30]}")
                
                pbar.update(1)
    
    # 打印统计
    print(f"\n" + "="*50)
    print(f"清理完成:")
    print(f"✅ 保留文件: {len(cleaned_files)}")
    print(f"❌ 删除文件: {len(removed_files)}")
    print(f"⚠️  错误文件: {len(error_files)}")
    
    if removed_files and len(removed_files) <= 20:
        print(f"\n删除的文件:")
        for f in removed_files:
            print(f"  - {f}")
    elif removed_files:
        print(f"\n删除的文件（前10个）:")
        for f in removed_files[:10]:
            print(f"  - {f}")
        print(f"  ... 还有{len(removed_files)-10}个")
    
    if error_files:
        print(f"\n处理出错的文件:")
        for f, err in error_files[:5]:
            print(f"  - {f}: {err}")
    
    return cleaned_files, removed_files, error_files


# 使用示例
if __name__ == "__main__":
    source = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track"
    target = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track_cleaned"
    
    # 使用8个进程并行处理
    cleaned, removed, errors = batch_clean_dataset_parallel(
        source_dir=source,
        target_dir=target,
        max_workers=32  # 根据CPU核心数调整
    )
    
    print(f"\n处理完成！")
    print(f"成功率: {len(cleaned)/(len(cleaned)+len(removed)+len(errors))*100:.1f}%")