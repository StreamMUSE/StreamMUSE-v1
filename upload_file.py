from huggingface_hub import HfApi

# --- 配置你的上传信息 ---
repo_id = "S-tanley/M2A"
local_folder_path = "/Users/zhengbowen/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/e2819ded95e6751ca55d773ffdf7a515/Message/MessageTemp/5bca8bd6e5be5cf9194667dfd69c5add/File/RT_Accompaniment/checkpoints" # 要上传的本地文件夹
path_in_repo = "./ModelLekai" # 在远程仓库中保存的路径

# --- 上传逻辑 ---
api = HfApi()

try:
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",  # "dataset" 或 "model"
        commit_message=f"Upload content of '{local_folder_path}' to '{path_in_repo}'",
    )
    print(f"✅ 文件夹 '{local_folder_path}' 已成功上传至 Hub 的 '{path_in_repo}'。")
except Exception as e:
    print(f"❌ 上传过程中发生错误: {e}")