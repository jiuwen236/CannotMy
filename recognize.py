import os
import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab, Image, ImageDraw
import torch
import torchvision.transforms
from sklearn.metrics.pairwise import cosine_similarity

# 先创建无预训练权重的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(weights=None)
weights_path = r'models\resnet18-5c106cde.pth'
state_dict = torch.load(weights_path,weights_only=False)
model.load_state_dict(state_dict)
model.eval().to(device)

transforms_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(117),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 配置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

# 鼠标交互全局变量
drawing = False
roi_box = []

# 预定义相对坐标
relative_regions_nums = [
    (0.0, 0.75, 0.155, 1),
    (0.155, 0.75, 0.28, 1),
    (0.28, 0.75, 0.39, 1),
    (0.61, 0.75, 0.72, 1),
    (0.72, 0.75, 0.84, 1),
    (0.84, 0.75, 0.95, 1)
]
relative_regions = [
    (0.0, 0.0, 0.1173, 1),
    (0.1220, 0.0, 0.2390, 1),
    (0.2451, 0.0, 0.3624, 1),
    (0.6359, 0.0, 0.7532, 1),
    (0.7593, 0.0, 0.8759, 1),
    (0.8824, 0.0, 1, 1)
]

ref_features = {}  # 全局变量用于存储预计算的特征

def save_number_image(number, processed, mon_id):
    """保存数字图片到对应文件夹
    Args:
        number: 识别出的数字
        processed: 处理后的图片
        mon_id: 怪物ID
    """
    if number and mon_id != 0:
        # 创建数字对应的文件夹
        num_folder = os.path.join("images", "nums", str(number))
        if not os.path.exists(num_folder):
            os.makedirs(num_folder)

        # 获取文件夹中已有的图片数量
        existing_files = [f for f in os.listdir(num_folder) if f.endswith('.png')]
        next_index = len(existing_files) + 1

        # 保存图片，命名为 id_序号.png
        save_path = os.path.join(num_folder, f"{mon_id}_{next_index}.png")
        cv2.imwrite(save_path, processed)


def mouse_callback(event, x, y, flags, param):
    global roi_box, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_box = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = param.copy()
        cv2.rectangle(img_copy, roi_box[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_box.append((x, y))
        drawing = False


def select_roi():
    """改进的交互式区域选择"""
    global roi_box  # 声明为全局变量
    while True:
        # 获取初始截图
        screenshot = np.array(ImageGrab.grab())
        img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # 添加操作提示
        cv2.putText(img, "Drag to select area | ENTER:confirm | ESC:retry",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加示例图片
        example_img = cv2.imread("images/eg.png")
        # 显示示例图片在单独的窗口中
        cv2.imshow("example", example_img)

        # 显示窗口
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720)
        cv2.setMouseCallback("Select ROI", mouse_callback, img)
        cv2.imshow("Select ROI", img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13 and len(roi_box) == 2:  # Enter确认
            # 标准化坐标 (x1,y1)为左上角，(x2,y2)为右下角
            x1, y1 = min(roi_box[0][0], roi_box[1][0]), min(roi_box[0][1], roi_box[1][1])
            x2, y2 = max(roi_box[0][0], roi_box[1][0]), max(roi_box[0][1], roi_box[1][1])
            return [(x1, y1), (x2, y2)]
        elif key == 27:  # ESC重试
            roi_box = []
            continue


def find_best_match(target):
    """改进后的特征匹配方法"""
    # 提取目标图片特征（使用与参考图片相同的预处理）
    target_pil = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    img_tensor = transforms_preprocess(target_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        target_features = model(img_tensor).cpu().numpy().flatten()

    # 与预存特征进行比对
    best_match_id = None
    max_similarity = -1
    for ref_id, ref_feature in ref_features.items():
        similarity = cosine_similarity([target_features], [ref_feature])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_id = ref_id
    # 用于测试匹配度的代码
    # print(f"best_id{best_match_id}max_similarity{max_similarity}")
    # 测出来空白地区对怪大约在0.5-相似度，怪的正常相似度大约在0.75~0.8+，阈值设定为0.65
    # 为啥还保留id=0的检测呢，问就是懒得改
    if (best_match_id != 0) & (max_similarity < 0.65):
        best_match_id = 0

    return best_match_id, max_similarity


def add_black_border(img, border_size=3):
    return cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # BGR格式的黑色
    )


def crop_to_min_bounding_rect(image):
    """裁剪图像到包含所有轮廓的最小外接矩形"""
    # 转为灰度图（如果传入的是二值图，这个操作不会有问题）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 寻找轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓就直接返回原图
    if not contours:
        return image

    # 合并所有轮廓点并获取外接矩形
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    # 裁剪图片并返回
    return image[y:y + h, x:x + w]


def preprocess(img):
    """彩色图像二值化处理，增强数字可见性"""
    # 检查图像是否为彩色
    if len(img.shape) == 2:
        # 如果是灰度图像，转换为三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 创建较宽松的亮色阈值范围（包括浅灰、白色等亮色）
    # BGR格式
    lower_bright = np.array([210, 210, 210])
    upper_bright = np.array([255, 255, 255])

    # 基于颜色范围创建掩码
    bright_mask = cv2.inRange(img, lower_bright, upper_bright)

    # 进行形态学操作，增强文本可见性
    # 创建一个小的椭圆形核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    # 膨胀操作，使文字更粗
    dilated = cv2.dilate(bright_mask, kernel, iterations=1)

    # 闭操作，填充文字内的小空隙
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    closed = dilated

    # 去除细小噪声：过滤不够大的连通区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1:
            # 用黑色填充宽度小于等于1的区域
            cv2.drawContours(closed, [contour], -1, 0, thickness=cv2.FILLED)
        if h <= 13:
            # 用黑色填充高度小于等于13的区域
            cv2.drawContours(closed, [contour], -1, 0, thickness=cv2.FILLED)

    return closed


def process_regions(main_roi, screenshot=None):
    """处理主区域中的所有区域（优化特征匹配）
    Args:
        main_roi: 主要感兴趣区域的坐标
        screenshot: 可选的预先捕获的截图
    Returns:
        区域处理结果的列表
    """
    results = []
    (x1, y1), (x2, y2) = main_roi
    main_width = x2 - x1
    main_height = y2 - y1

    # 如果没有提供screenshot，则获取最新截图（仅截取主区域）
    if screenshot is None:
        screenshot = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    else:
        # 从当前screenshot中提取主区域
        screenshot = screenshot[y1:y2, x1:x2]

    # 遍历所有区域
    for idx, rel in enumerate(relative_regions):
        try:
            # ================== 模板匹配部分 ==================
            # 计算模板匹配的子区域坐标
            rx1 = int(rel[0] * main_width)
            ry1 = int(rel[1] * main_height)
            rx2 = int(rel[2] * main_width)
            ry2 = int(rel[3] * main_height)

            # 提取模板匹配用的子区域
            sub_roi = screenshot[ry1:ry2, rx1:rx2]

            # 使用预存特征进行匹配
            matched_id, confidence = find_best_match(sub_roi)

            # ================== OCR数字识别部分 ==================
            rel_num = relative_regions_nums[idx]
            rx1_num = int(rel_num[0] * main_width)
            ry1_num = int(rel_num[1] * main_height)
            rx2_num = int(rel_num[2] * main_width)
            ry2_num = int(rel_num[3] * main_height)

            # 提取OCR识别用的子区域
            number_roi = screenshot[ry1_num:ry2_num, rx1_num:rx2_num]

            # 预处理流程
            processed = preprocess(number_roi)
            processed = crop_to_min_bounding_rect(processed)  # 裁剪至外接矩形
            processed = add_black_border(processed, border_size=3)  # 加黑框

            # OCR识别（保留优化后的处理逻辑）
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789x×X'
            number = pytesseract.image_to_string(processed, config=custom_config).strip()
            number = number.replace('×', 'x').lower()  # 统一符号

            # 提取有效数字部分
            x_pos = number.find('x')
            if x_pos != -1:
                number = number[x_pos + 1:]  # 截取x之后的字符串
            number = ''.join(filter(str.isdigit, number))

            # 保存数据到结果集
            results.append({
                "region_id": idx,
                "matched_id": matched_id,
                "number": number if number else "N/A",
                "confidence": round(confidence, 2)
            })

            # 保存样本到训练集（根据需求开启）
            if number and matched_id != 0:
                save_number_image(number, processed, matched_id)

        except Exception as e:
            print(f"区域{idx}处理失败: {str(e)}")
            results.append({
                "region_id": idx,
                "error": str(e)
            })

    return results


def load_ref_images(ref_dir="images/ref"):
    """加载参考图片库并预计算特征"""
    global ref_features
    ref_features = {}

    for i in range(35):
        path = os.path.join(ref_dir, f"{i}.png")
        if os.path.exists(path):
            # 读取并预处理图片
            img = cv2.imread(path)
            if img is None:
                continue

            # 转换为PIL Image并进行预处理
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # 使用相同的预处理流程
            img_tensor = transforms_preprocess(img_pil).unsqueeze(0).to(device)

            # 提取特征
            with torch.no_grad():
                features = model(img_tensor).cpu().numpy()

            # 存储特征向量
            ref_features[i] = features.flatten()  # 展平特征向量便于后续计算


load_ref_images(ref_dir="images/ref") # 初始化参考图片的特征向量


if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域...")
    main_roi = select_roi()
    ref_images = load_ref_images()
    results = process_regions(main_roi)
    # 输出结果
    print("\n识别结果：")
    for res in results:
        if 'error' in res:
            print(f"区域{res['region_id']}: 错误 - {res['error']}")
        else:
            if res['matched_id'] != 0:
                print(
                    f"区域{res['region_id']} => 匹配ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")
