import numpy as np

# 你的 npz 文件路径
file_path = 'data/PEMS08/PEMS08.npz'

# 加载 npz 文件
data = np.load(file_path)

# 打印所有字段名（keys）
print("Keys in npz file:", list(data.keys()))
#.\.venv\Scripts\python.exe -m pip install -r requirements.txt（用来安装依赖的命令，下载到虚拟幻境里就不会污染全局环境了）
