from huggingface_hub import snapshot_download
import os

# 能够下载指定repo的model/dataset
def download_repo(repo_id: str, local_dir_prefix: str, max_workers: int = 4):
    local_dir = repo_id.split("/")[-1]
    local_dir = os.path.join(local_dir_prefix, local_dir)
    
    # 如果目录已存在，直接返回，不触发下载
    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists, skipping download")
        return local_dir
    
    # 缓存到当前目录会造成CloudIDE卡死
    print(f"download start to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    while True:
        try:
            # 下载文件
            # force_download为True会强制重新下载已完成的文件，中断的文件即使设为False，重新下载时也能从.cache中重新下载，下载过程管理在.cache中进行，不需要设为True
            # etag_timeout增加到300后，不再出现timeout
            downloaded_file = snapshot_download(repo_id=repo_id, 
                local_dir=local_dir, 
                etag_timeout=300,
                force_download=False,
                max_workers=max_workers,
            )
            print(f"download finish finally: {downloaded_file}")
            break
        except OSError as e:
            print(f"download error: {e}")
            continue

    return downloaded_file


        

if __name__ == "__main__":
    # download_repo("Qwen/QwQ-32B", "/opt/tiger")
    # download_repo("simplescaling/s1K-1.1", "./")
    # download_repo("Qwen/Qwen3-14B", "/opt/tiger")
    # download_repo("lmsys/gpt-oss-20b-bf16", "/opt/tiger")
    download_repo("Qwen/Qwen3-VL-30B-A3B-Instruct", "/mnt/bn/strategy-mllm-train/user/hjy/base_models")
    # download_repo("openai/gpt-oss-20b", "/opt/tiger")
    # download_repo("Qwen/Qwen3-4B-Instruct-2507", "/opt/tiger")
    # download_repo("Qwen/Qwen3-30B-A3B-Instruct-2507", "/opt/tiger")
    # download_repo("Qwen/Qwen3-30B-A3B-Thinking-2507", "/opt/tiger")