import os
from functools import cache

import numpy as np
import torch

from recognize import MONSTER_COUNT


@cache
def get_device(prefer_gpu=True):
    """
    prefer_gpu (bool): 是否优先尝试使用GPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
            return torch.device("xpu")
    return torch.device("cpu")


device = get_device()


class CannotModel:
    def __init__(self, model_path="models/best_model_full.pth"):
        self.device = get_device()
        self.model_path = model_path  # 存储模型路径
        self.model = None  # 模型实例
        self.load_model()  # 初始化时加载模型


    def load_model(self):
        """初始化时加载模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    rf"未找到训练好的模型文件 {self.model_path}，请先训练模型"
                )

            try:
                model = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load(
                    self.model_path, map_location=self.device
                )
            model.eval()
            self.model = model.to(self.device)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            raise e  # 无法继续运行，退出程序
        
    def export_onnx(self,outputpath, monster_count=MONSTER_COUNT):
        # 确保模型在 CPU 上（避免设备不一致）
        self.model = self.model.cpu()
        self.model.eval()

        # 生成虚拟输入（与模型同设备）
        device = next(self.model.parameters()).device
        dummy_left_counts = torch.randint(0, 10, (1, monster_count), dtype=torch.int16, device=device)
        dummy_right_counts = torch.randint(0, 10, (1, monster_count), dtype=torch.int16, device=device)

        # 获取符号和绝对值张量（确保在相同设备）
        left_signs = torch.sign(dummy_left_counts.to(torch.int64)).to(device)
        left_counts = torch.abs(dummy_left_counts.to(torch.int64)).to(device)
        right_signs = torch.sign(dummy_right_counts.to(torch.int64)).to(device)
        right_counts = torch.abs(dummy_right_counts.to(torch.int64)).to(device)

        # 导出参数
        input_names = ["left_signs", "left_counts", "right_signs", "right_counts"]
        dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
        dynamic_axes["output"] = {0: 'batch_size'}

        # 导出 ONNX
        torch.onnx.export(
            self.model,
            (left_signs, left_counts, right_signs, right_counts),
            outputpath,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=20,
            verbose=True  # 开启详细输出便于调试
        )

    def get_prediction(self, left_counts: np.typing.ArrayLike, right_counts: np.typing.ArrayLike):
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        # 转换为张量并处理符号和绝对值
        left_signs = (
            torch.sign(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        left_counts = (
            torch.abs(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_signs = (
            torch.sign(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_counts = (
            torch.abs(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )

        # 预测流程
        with torch.no_grad():
            # 使用修改后的模型前向传播流程
            prediction = self.model(
                left_signs, left_counts, right_signs, right_counts
            ).item()

            # 确保预测值在有效范围内
            if np.isnan(prediction) or np.isinf(prediction):
                print("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5

            # 检查预测结果是否在[0,1]范围内
            if prediction < 0 or prediction > 1:
                prediction = max(0, min(1, prediction))

        return prediction
