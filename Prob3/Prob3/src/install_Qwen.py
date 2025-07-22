from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3-32B', cache_dir='/autodl-pub')
print("模型已下载到：", model_dir)