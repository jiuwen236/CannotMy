import onnxruntime as ort
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CannotModel:
    def __init__(self,model_path = "models/best_model_full.onnx"):
        self.session = None  # ONNX Runtime 会话
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """加载 ONNX 模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"未找到 ONNX 模型文件 {self.model_path}")
            
            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建会话（默认使用 CPU）
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
        except Exception as e:
            raise RuntimeError(f"ONNX 模型加载失败: {str(e)}")

    def get_prediction(self, left_counts: np.ndarray, right_counts: np.ndarray):
        if self.session is None:
            raise RuntimeError("模型未正确初始化")
        
        def validate_input(arr):
            """验证并转换输入数据"""
            # 转换为 int64 类型
            arr = arr.astype(np.int64)
    
            # 添加批次维度（如果输入是单样本）
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # shape: (1, 56)
            return arr
        
        # 添加批次维度
        inputs = {
            "left_counts": validate_input(left_counts).astype(np.int64),
            "right_counts": validate_input(right_counts).astype(np.int64)
        }
        
        # 执行推理
        try:
            output = self.session.run(
                output_names=["output"],
                input_feed=inputs
            )
            print(output)
            prediction = output[0]
        except Exception as e:
            raise RuntimeError(f"推理失败: {str(e)}")
        
        # 后处理（与原逻辑一致）
        if np.isnan(prediction) or np.isinf(prediction):
            logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
            prediction = 0.5
        
        prediction = np.clip(prediction, 0.0, 1.0)
        return float(prediction)