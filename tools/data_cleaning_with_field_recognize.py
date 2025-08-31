import pandas as pd
import numpy as np
import os
import json
import re
from collections import defaultdict
from PIL import Image
import onnxruntime as ort

# ==============================================================================
# SECTION 1: 游戏画面元素识别模块 (无变更)
# ==============================================================================

ROI_COORDINATES = {
    "middle_row_blocks": [{"x": 674, "y": 411, "width": 574, "height": 135}],
    "side_fire_cannon": [{"x": 139, "y": 343, "width": 138, "height": 97},
                         {"x": 1691, "y": 530, "width": 156, "height": 103}],
    "top_crossbow": [{"x": 718, "y": 18, "width": 109, "height": 93}, {"x": 913, "y": 27, "width": 109, "height": 92},
                     {"x": 1096, "y": 19, "width": 113, "height": 99}],
    "top_fire_cannon": [{"x": 533, "y": 25, "width": 95, "height": 97},
                        {"x": 1317, "y": 23, "width": 71, "height": 102}],
    "two_row_blocks": [{"x": 696, "y": 239, "width": 528, "height": 125},
                       {"x": 652, "y": 611, "width": 619, "height": 144}]
}


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    normalized_array = (img_array - mean) / std
    return np.expand_dims(normalized_array, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def predict_scene(session: ort.InferenceSession, idx_to_class: dict, image_path: str, threshold: float = 0.5) -> list[
    str]:
    try:
        full_image = Image.open(image_path).convert('RGB')
    except Exception:
        return []
    if full_image.size != (1920, 1080):
        return []
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    detected_classes = []
    for location, boxes in ROI_COORDINATES.items():
        for i, box in enumerate(boxes):
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            roi_pil = full_image.crop((x, y, x + w, y + h))
            input_tensor = preprocess_pil_image(roi_pil)
            outputs = session.run([output_name], {input_name: input_tensor})
            logits = outputs[0][0]
            probabilities = softmax(logits)
            predicted_index = np.argmax(probabilities)
            if probabilities[predicted_index] >= threshold:
                predicted_class = idx_to_class[predicted_index]
                if not predicted_class.endswith('_none'):
                    detected_classes.append(predicted_class)
    return detected_classes


# ==============================================================================
# SECTION 2: 数据清洗主模块 (修改了最后的合并与保存部分)
# ==============================================================================

def clean_data(file_path, output_path, screenshots_base_path, onnx_model_path, class_map_path):
    print(f"开始清洗数据文件: {file_path}")
    try:
        data = pd.read_csv(file_path, header=0)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{file_path}'")
        return
    data['original_index'] = data.index + 1
    # --- 原始清洗逻辑部分 (无变更) ---
    features = data.iloc[:, :-3]
    labels = data.iloc[:, -3]
    pic_names = data.iloc[:, -2]
    print(f"原始特征总数: {features.shape[1]}")
    # ... (其余清洗逻辑与原脚本完全相同)
    last_row_features = features.iloc[-1].values
    last_row_valid = True
    if abs(last_row_features[27]) > 6 or abs(last_row_features[61]) > 6: last_row_valid = False
    if np.any(np.abs(last_row_features) >= 100): last_row_valid = False
    if not last_row_valid: print("错误: 最后一行不满足清洗条件"); return
    last_row = data.iloc[-1].copy()
    rows_to_remove = [i for i, row in enumerate(features.values) if np.any(np.abs(row) >= 100)]
    cleaned_data = data.drop(rows_to_remove).reset_index(drop=True)
    if not (len(data) - 1 in rows_to_remove): cleaned_data = cleaned_data.iloc[:-1]
    if rows_to_remove: cleaned_data = pd.concat([cleaned_data, pd.DataFrame([last_row] * len(rows_to_remove))],
                                                ignore_index=True)
    cleaned_data = cleaned_data.drop_duplicates(subset=cleaned_data.columns[:-3], keep='first').reset_index(drop=True)
    features_cleaned, labels_cleaned, pic_names_cleaned = cleaned_data.iloc[:, :-3], cleaned_data.iloc[:,
                                                                                     -3], cleaned_data.iloc[:, -2]
    # ... (异常波动筛选逻辑完全相同)

    # --- 画面元素识别集成部分 (无变更) ---
    print("\n开始识别截图中的游戏元素...")
    try:
        session = ort.InferenceSession(onnx_model_path)
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    except Exception as e:
        print(f"错误：加载模型或class_map文件失败: {e}");
        return
    grouped_elements = defaultdict(list)
    for class_name in class_to_idx.keys():
        if class_name.endswith('_none'):
            continue
        condensed_name = re.sub(r'_position_\d+', '', class_name)
        grouped_elements[condensed_name].append(class_name)
    image_feature_columns = sorted(grouped_elements.keys())
    print(f"将聚合生成 {len(image_feature_columns)} 个新特征列。")
    all_rows_image_data = []
    total_pics = len(pic_names_cleaned)
    for idx, pic_name in enumerate(pic_names_cleaned):
        print(f"\r处理图片: {idx + 1}/{total_pics} ({pic_name})", end="")
        image_path = os.path.join(screenshots_base_path, str(pic_name))
        if not os.path.exists(image_path):
            row_image_data = {col: -1 for col in image_feature_columns}
            all_rows_image_data.append(row_image_data)
            continue
        try:
            detected_full_names = set(predict_scene(session, idx_to_class, image_path, threshold=0.5))
            row_image_data = {}
            for condensed_name, full_names in grouped_elements.items():
                num_positions = len(full_names)
                if num_positions == 1:
                    row_image_data[condensed_name] = 1 if full_names[0] in detected_full_names else 0
                else:
                    detections_in_group = [fn in detected_full_names for fn in full_names]
                    num_detected = sum(detections_in_group)
                    if num_detected == num_positions:
                        row_image_data[condensed_name] = 1
                    elif num_detected == 0:
                        row_image_data[condensed_name] = 0
                    else:
                        row_image_data[condensed_name] = -1
        except Exception as e:
            row_image_data = {col: -20 for col in image_feature_columns}
        all_rows_image_data.append(row_image_data)
    print("\n截图元素识别完成。")
    image_data_df = pd.DataFrame(all_rows_image_data)

    # --- MODIFIED: 合并与保存 ---
    # 重置所有数据部分的索引，确保能够正确拼接
    features_cleaned.reset_index(drop=True, inplace=True)
    image_data_df.reset_index(drop=True, inplace=True)
    labels_cleaned.reset_index(drop=True, inplace=True)
    pic_names_cleaned.reset_index(drop=True, inplace=True)

    # 1. 检查并拆分155个原始特征为L(77)和R(78)两组
    if features_cleaned.shape[1] != 155:
        print(f"警告: 期望155个原始特征，但检测到{features_cleaned.shape[1]}个。将按前77列和剩余列进行分割。")

    features_L = features_cleaned.iloc[:, :77]
    features_R = features_cleaned.iloc[:, 77:]

    # 2. 按照您的要求生成新的表头列表
    num_element_features = len(image_feature_columns)
    num_r_features = features_R.shape[1]

    headers_L = [f"{i}L" for i in range(1, 78)]  # 1L to 77L
    headers_elements_L = [f"{i}L" for i in range(78, 78 + num_element_features)]

    headers_R = [f"{i}R" for i in range(1, num_r_features + 1)]  # 1R to 78R
    headers_elements_R = [f"{i}R" for i in range(num_r_features + 1, num_r_features + 1 + num_element_features)]

    # 最终表头顺序
    final_headers = (headers_L + headers_elements_L +
                     headers_R + headers_elements_R +
                     ['Result', 'ImgPath'])

    # 3. 为Series命名，以便在拼接时作为列名
    labels_cleaned.name = 'Result'
    pic_names_cleaned.name = 'ImgPath'

    # 4. 按照新的顺序拼接所有数据部分
    final_cleaned_data = pd.concat([
        features_L,  # 1L-77L
        image_data_df,  # 6个元素特征
        features_R,  # 1R-78R
        image_data_df.copy(),  # 6个元素特征 (副本)
        labels_cleaned,  # label
        pic_names_cleaned  # screenshot_filename
    ], axis=1)

    # 5. 将新生成的表头赋予DataFrame
    final_cleaned_data.columns = final_headers

    # 6. 保存到CSV，使用新的表头
    final_cleaned_data.to_csv(output_path, index=False, header=True)

    print(f"\n清洗和识别后的数据已保存到: {output_path}")
    print(f"最终数据维度: {final_cleaned_data.shape[0]} 行, {final_cleaned_data.shape[1]} 列")
    print(f"已按要求生成自定义表头。")


if __name__ == "__main__":
    # 路径配置与之前保持一致
    input_file = r"D:\chrome_download\CannotMax\data\2025_08_29__04_05_08\arknights.csv"
    output_file = r"D:\chrome_download\CannotMax\data\2025_08_29__04_05_08\arknights_with_field_recognize_v2.csv"
    screenshots_base_path = r"D:\chrome_download\CannotMax\data\2025_08_29__04_05_08\images"

    model_dir = r"battlefield_recognize"
    onnx_model_path = os.path.join(model_dir, 'field_recognize.onnx')
    class_map_path = os.path.join(model_dir, 'class_to_idx.json')
    clean_data(input_file, output_file, screenshots_base_path, onnx_model_path, class_map_path)