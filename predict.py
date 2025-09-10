import re
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import torch
import logging

from config import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)

def get_device(prefer_gpu=True):
    """
    prefer_gpu (bool): 是否优先尝试使用GPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            logger.info("Use torch with cuda")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Use torch with mps")
            return torch.device("mps")  # Apple Silicon GPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
            logger.info("Use torch with xpu")
            return torch.device("xpu")
    logger.info("Use torch with cpu")
    return torch.device("cpu")

class CannotModel:
    def __init__(self, model_path="models"):
        self.device = get_device()
        # 收集模型文件（非递归）
        self.model_paths = self._resolve_model_paths(model_path)
        # 向后兼容：main.py 里会读取 model_path 来展示，这里放一个可读标签
        if Path(model_path).is_dir():
            self.model_path = f"{Path(model_path).name} ({len(self.model_paths)} models)"
        else:
            # 可能是单文件
            self.model_path = str(model_path)
        self.models = []
        self.model = None  # 兼容旧用法（如导出 onnx）
        try:
            self.load_model()  # 初始化时加载模型（现在会加载所有模型）
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.models = []
            self.model = None

    def _resolve_model_paths(self, path):
        """返回需要加载的模型文件列表。
        - 若为目录：返回该目录下所有 .pth 文件（非递归）。
        - 若为文件：返回该文件。
        - 其他：返回空列表。
        """
        p = Path(path)
        if p.is_dir():
            models = [str(f) for f in p.iterdir() if f.is_file() and f.suffix == ".pth"]
            models.sort(key=lambda s: Path(s).name)
            if not models:
                logger.error(f"目录中未找到模型文件: {p}")
            else:
                logger.info(f"共找到 {len(models)} 个模型: {[Path(m).name for m in models]}")
            return models
        if p.is_file():
            logger.info(f"使用指定模型文件: {p}")
            return [str(p)]
        logger.error(f"提供的模型路径无效: {path}")
        return []

    def load_model(self):
        """加载所有模型到内存。"""
        if not self.model_paths:
            raise FileNotFoundError("未找到任何模型文件，请先训练或放置 .pth 模型到 models/ 目录")
        loaded = []
        for mp in self.model_paths:
            if not Path(mp).exists():
                logger.warning(f"模型文件不存在，跳过: {mp}")
                continue
            try:
                try:
                    m = torch.load(mp, map_location=self.device, weights_only=False)
                except TypeError:
                    m = torch.load(mp, map_location=self.device)
                m.eval()
                loaded.append(m.to(self.device))
                logger.info(f"已加载模型: {Path(mp).name}")
            except Exception as e:
                logger.error(f"加载模型失败 {mp}: {e}")
        if not loaded:
            raise RuntimeError("没有成功加载的模型")
        self.models = loaded
        self.model = self.models[0]  # 兼容旧逻辑（例如 export_onnx 使用）
        
    def export_onnx(self,outputpath, monster_count=MONSTER_COUNT):
        # 确保模型在 CPU 上（避免设备不一致）
        if self.model is None:
            raise RuntimeError("没有可用模型用于导出")
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
        if not self.models:
            raise RuntimeError("模型未正确初始化")

        # 转换为张量并处理符号和绝对值
        left_signs = (
            torch.sign(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        left_counts_t = (
            torch.abs(torch.tensor(left_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_signs = (
            torch.sign(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )
        right_counts_t = (
            torch.abs(torch.tensor(right_counts, dtype=torch.int16))
            .unsqueeze(0)
            .to(self.device)
        )

        probs = []
        USE_LOGITS = True
        with torch.no_grad():
            for mp, m in zip(self.model_paths, self.models):
                try:
                    out = m(left_signs, left_counts_t, right_signs, right_counts_t)
                    # 使用 logits -> sigmoid（与原逻辑保持一致，可通过 USE_LOGITS 控制）
                    if USE_LOGITS:
                        prob = torch.sigmoid(out).item()
                    else:
                        prob = out.item()
                except Exception as e:
                    logger.error(f"前向推理失败 {Path(mp).name}: {e}")
                    continue
                # 容错与裁剪
                if np.isnan(prob) or np.isinf(prob):
                    logger.warning(f"模型 {Path(mp).name} 预测为 NaN/Inf，记为 0.5")
                    prob = 0.5
                prob = max(0.0, min(1.0, prob))
                probs.append(prob)
                logger.info(f"{Path(mp).name} -> 右方胜率: {prob:.4f} (左方 {1-prob:.4f})")

        avg = float(np.mean(probs)) if probs else 0.5
        return avg

    def get_prediction_with_terrain(self, full_features: np.typing.ArrayLike):
        """使用包含地形特征的完整特征向量进行预测"""
        if not self.models:
            raise RuntimeError("模型未正确初始化")

        # 检查特征向量长度
        expected_length = MONSTER_COUNT * 2 + FIELD_FEATURE_COUNT * 2  # 77L + 6L + 77R + 6R = 166
        if len(full_features) != expected_length:
            logger.warning(f"特征向量长度不匹配: 期望{expected_length}, 实际{len(full_features)}，回退到无地形预测")
            left_counts = full_features[:MONSTER_COUNT]
            right_counts = full_features[MONSTER_COUNT:MONSTER_COUNT*2]
            return self.get_prediction(left_counts, right_counts)

        # 提取各个部分
        left_monsters = full_features[:MONSTER_COUNT]  # 1L-77L
        left_terrain = full_features[MONSTER_COUNT:MONSTER_COUNT+FIELD_FEATURE_COUNT]  # 78L-83L
        right_monsters = full_features[MONSTER_COUNT+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT]  # 1R-77R
        right_terrain = full_features[MONSTER_COUNT*2+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT*2]  # 78R-83R

        # 转换为张量并处理符号和绝对值
        # 对于怪物特征，使用符号和绝对值
        # 对于地形特征，不需要符号处理（地形特征本身就是0/1值）
        left_monster_signs = torch.sign(torch.tensor(left_monsters, dtype=torch.int16))
        left_terrain_signs = torch.ones_like(torch.tensor(left_terrain, dtype=torch.int16))  # 地形特征符号为1
        left_signs = torch.cat([left_monster_signs, left_terrain_signs]).unsqueeze(0).to(self.device)

        left_monster_counts = torch.abs(torch.tensor(left_monsters, dtype=torch.int16))
        left_terrain_counts = torch.tensor(left_terrain, dtype=torch.int16)  # 地形特征直接使用原值
        left_counts_tensor = torch.cat([left_monster_counts, left_terrain_counts]).unsqueeze(0).to(self.device)

        right_monster_signs = torch.sign(torch.tensor(right_monsters, dtype=torch.int16))
        right_terrain_signs = torch.ones_like(torch.tensor(right_terrain, dtype=torch.int16))  # 地形特征符号为1
        right_signs = torch.cat([right_monster_signs, right_terrain_signs]).unsqueeze(0).to(self.device)

        right_monster_counts = torch.abs(torch.tensor(right_monsters, dtype=torch.int16))
        right_terrain_counts = torch.tensor(right_terrain, dtype=torch.int16)  # 地形特征直接使用原值
        right_counts_tensor = torch.cat([right_monster_counts, right_terrain_counts]).unsqueeze(0).to(self.device)

        probs = []
        USE_LOGITS = True
        print(f"USE_LOGITS: {USE_LOGITS}")
        with torch.no_grad():
            for mp, m in zip(self.model_paths, self.models):
                try:
                    out = m(left_signs, left_counts_tensor, right_signs, right_counts_tensor)
                    if USE_LOGITS:
                        prob = torch.sigmoid(out).item()
                    else:
                        prob = out.item()
                except Exception as e:
                    logger.error(f"前向推理失败 {Path(mp).name}: {e}")
                    continue
                if np.isnan(prob) or np.isinf(prob):
                    logger.warning(f"模型 {Path(mp).name} 预测为 NaN/Inf，记为 0.5")
                    prob = 0.5
                prob = max(0.0, min(1.0, prob))
                probs.append(prob)
                logger.info(f"{Path(mp).name} -> 右方胜率: {prob:.4f} (左方 {1-prob:.4f})")

        avg = float(np.mean(probs)) if probs else 0.5
        return avg
