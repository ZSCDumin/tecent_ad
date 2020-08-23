from yacs.config import CfgNode
import os


_C = CfgNode()

_C.train_dir = "D:/Data/tencent2020/train_preliminary/"
_C.test_dir = "D:/Data/tencent2020/test/"
_C.output = "D:/Data/tencent2020/output/"
_C.decode = "D:/Data/tencent2020/output/decode/"
_C.features = "D:/Data/tencent2020/output/features/"
os.makedirs(_C.decode, exist_ok=True)
os.makedirs(_C.features, exist_ok=True)
