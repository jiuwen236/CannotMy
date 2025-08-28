from predict import *
from train import *
import predict_onnx



model_path = "models/best_model_full.pth"

# TODO: 怪物数量需改为使用recognize中定义的常量

def replace_suffix(s):
    idx = s.rfind('.')
    if idx == -1:
        return s + '.onnx'
    else:
        return s[:idx] + '.onnx'
    
output_path = replace_suffix(model_path)

model = CannotModel(model_path)
model.load_model()  # 加载原始 PyTorch 模型
model.export_onnx(output_path)  # 导出 ONNX 模型

raw_data = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]
#验证导出结果

data_array = np.array(raw_data)
left = data_array[:56]
right = data_array[56:]

Onnxmodel = predict_onnx.CannotModel(model_path = output_path)
prediction = Onnxmodel.get_prediction(left, right)
print("预测结果:", prediction)
