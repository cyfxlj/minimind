# Dataset Download
# 数据集中存在多种列格式（conversations / chosen+rejected / text），
# 不能一次性 MsDataset.load，需按格式分组下载并分别加载。

import os
from datasets import load_dataset

# 若之前加载失败，可先删掉不完整缓存再运行（避免 OSError 39）:
# rm -rf ./dataset/gongjy___minimind_dataset ./downloads

DATASET_ID = "gongjy/minimind_dataset"
# 数据集 API（与 MsDataset 使用的地址一致；snapshot_download 对 dataset 可能 404）
REPO_URL = "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "minimind_raw")

# 按列格式分组（与仓库中文件名对应）
CONVERSATIONS_FILES = [
    "pretrain_hq.jsonl"
    # "lora_identity.jsonl",
    # "lora_medical.jsonl",
    # "sft_512.jsonl",
    # "sft_1024.jsonl",
    # "sft_2048.jsonl",
    # "sft_mini_512.jsonl",
    # "r1_mix_1024.jsonl",
    # "rlaif-mini.jsonl",
]
DPO_FILES = ["dpo.jsonl"]
PRETRAIN_FILES = ["pretrain_hq.jsonl"]


def _download_file(file_name: str) -> str:
    """从 ModelScope 数据集 API 下载单个文件到 LOCAL_DIR。"""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    path = os.path.join(LOCAL_DIR, file_name)
    if os.path.isfile(path):
        return path
    try:
        import requests
        r = requests.get(
            REPO_URL,
            params={"Source": "SDK", "Revision": "master", "FilePath": file_name},
            stream=True,
            timeout=60,
        )
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return path
    except Exception as e:
        if os.path.isfile(path):
            os.remove(path)
        raise RuntimeError(f"Download {file_name} failed: {e}") from e


def _ensure_downloaded():
    """下载数据集到本地目录。优先用 API 按文件下载（dataset 用 snapshot_download 易 404）。"""
    all_files = CONVERSATIONS_FILES + DPO_FILES + PRETRAIN_FILES
    os.makedirs(LOCAL_DIR, exist_ok=True)
    for f in all_files:
        p = os.path.join(LOCAL_DIR, f)
        if not os.path.isfile(p):
            _download_file(f)
    return LOCAL_DIR


def load_by_format(format_name="conversations", cache_dir=None):
    """
    按格式加载，避免列冲突。
    format_name: "conversations" | "dpo" | "pretrain"
    """
    _ensure_downloaded()
    if format_name == "conversations":
        files = [os.path.join(LOCAL_DIR, f) for f in CONVERSATIONS_FILES]
    elif format_name == "dpo":
        files = [os.path.join(LOCAL_DIR, f) for f in DPO_FILES]
    elif format_name == "pretrain":
        files = [os.path.join(LOCAL_DIR, f) for f in PRETRAIN_FILES]
    else:
        raise ValueError("format_name must be one of: conversations, dpo, pretrain")
    exist = [p for p in files if os.path.isfile(p)]
    if not exist:
        raise FileNotFoundError(f"No files found for {format_name}: {files}")
    return load_dataset("json", data_files=exist, split="train", cache_dir=cache_dir)


if __name__ == "__main__":
    # 若曾因列不一致报错，请先删不完整缓存再运行:
    #   rm -rf dataset/gongjy___minimind_dataset dataset/downloads
    _ensure_downloaded()
    ds = load_by_format("conversations", cache_dir=os.path.dirname(LOCAL_DIR))
    print("conversations:", len(ds), "columns:", ds.column_names)
