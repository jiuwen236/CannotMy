import subprocess
import time
import cv2
import numpy as np
import logging
import gzip


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AdbConnector:
    def __init__(self):
        self.adb_path = r".\platform-tools\adb.exe"
        # 默认设备序列号，可以在main.py中修改
        self.manual_serial = "127.0.0.1:5555"
        self.screen_width = 0
        self.screen_height = 0
        self.device_serial = ""
        self.is_connected = False

    def connect(self):
        # 初始化设备序列号
        try:
            self.device_serial = self.get_device_serial()
            logger.info(f"最终使用设备: {self.device_serial}")
        except RuntimeError as e:
            logger.exception(f"初始化设备序列号错误: ", e)
            exit(1)

        if self.device_serial:
            self.connect_to_emulator()
            # 获取屏幕分辨率
            self.screen_width, self.screen_height = self.get_window_size()
        else:
            logger.warning(f"连接模拟器失败，使用默认分辨率1920x1080。")
            self.screen_width, self.screen_height = 1920, 1080

    def connect_to_emulator(self):
        try:
            # 使用绝对路径连接到雷电模拟器
            connect_cmd = f"{self.adb_path} connect {self.device_serial}"
            subprocess.run(connect_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.exception(f"ADB connect command failed: {e}")
        except FileNotFoundError as e:
            logger.exception(
                f"Error: {e}. Please ensure adb is installed and added to the system PATH."
            )

    def get_window_size(self):
        try:
            # 执行ADB命令获取分辨率
            size_cmd = f"{self.adb_path} -s {self.device_serial} shell wm size"
            result = subprocess.run(
                size_cmd, shell=True, capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()

            # 解析分辨率输出
            if "Physical size:" in output:
                res_str = output.split("Physical size: ")[1]
            elif "Override size:" in output:
                res_str = output.split("Override size: ")[1]
            else:
                raise ValueError("无法解析分辨率输出格式")

            # 分割分辨率并转换为整数
            width, height = map(int, res_str.split("x"))
            if width > height:
                global screen_width, screen_height
                screen_width = width
                screen_height = height
            else:
                screen_width = height
                screen_height = width
            logger.info(f"成功获取模拟器分辨率: {screen_width}x{screen_height}")
        except Exception as e:  # 否则使用默认分辨率
            logger.exception(f"获取分辨率失败，使用默认分辨率1920x1080。错误: {e}")
            screen_width = 1920
            screen_height = 1080
        return screen_width, screen_height

    def set_device_serial(self, serial):
        self.manual_serial = serial

    def get_device_serial(self):
        try:
            # 使用当前的manual_serial值
            if self.manual_serial == "":
                logger.error(f"当前manual_serial为空")
            connect_cmd = f"{self.adb_path} connect {self.manual_serial}"
            subprocess.run(connect_cmd, shell=True, check=True)

            # 检查手动设备是否在线
            device_cmd = f"{self.adb_path} devices"
            result = subprocess.run(
                device_cmd, shell=True, capture_output=True, text=True, timeout=5
            )

            devices = []
            for line in result.stdout.split("\n"):
                if "\tdevice" in line:
                    dev = line.split("\t")[0]
                    devices.append(dev)
                    if dev == self.manual_serial:
                        device_serial = dev
                        return dev

            # 自动选择第一个可用设备
            if devices:
                device_serial = devices[0]
                logger.info(f"自动选择设备: {device_serial}")
                return device_serial

            logger.warning("未找到可连接的Android设备")
            return None

        except Exception as e:
            logger.exception(f"设备检测失败", e)
            return None

    def capture_screenshot(self):
        return self.capture_screenshot_raw_gzip()

    def capture_screenshot_png(self):
        try:
            ta = time.time()
            # 获取二进制图像数据
            get_png_cmd = f"{self.adb_path} -s {self.device_serial} exec-out screencap -p"
            screenshot_data = subprocess.check_output(get_png_cmd, shell=True)
            # 将二进制数据转换为numpy数组
            img_array = np.frombuffer(screenshot_data, dtype=np.uint8)
            # 使用OpenCV解码图像
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图像数据")
            logger.debug(f"获取图片用时{time.time()-ta:.3f}s")
            return img
        except subprocess.CalledProcessError as e:
            logger.exception(f"Screenshot capture failed: {e}")
            return None
        except Exception as e:
            logger.exception(f"Image processing error: {e}")
            return None

    def decode_raw(self, data: bytes):
        if len(data) < 8:
            raise RuntimeError("RAW image is empty")
        width = data[0] << 0 | data[1] << 8 | data[2] << 16 | data[3] << 24
        height = data[4] << 0 | data[5] << 8 | data[6] << 16 | data[7] << 24
        if width != self.screen_width or height != self.screen_height:
            logger.error(
                f"width: {width} height: {height} != screen_width: {self.screen_width} screen_height: {self.screen_height}"
            )
            raise RuntimeError(f"RAW图像分辨率与屏幕分辨率不符")
        # 12 or 16. ref: https://android.googlesource.com/platform/frameworks/base/+/26a2b97dbe48ee45e9ae70110714048f2f360f97%5E%21/cmds/screencap/screencap.cpp
        std_size = 4 * width * height
        header_size = len(data) - std_size
        # 将二进制数据转换为numpy数组
        argb_array = np.frombuffer(data, dtype=np.uint8)[header_size:]

        # 确保数据长度正确（1920x1080分辨率，4通道）
        if len(argb_array) != 1920 * 1080 * 4:
            raise ValueError("Invalid data length for 1920x1080 ARGB image")

        # 转换为正确的形状 (高度, 宽度, 通道)
        argb_array = argb_array.reshape((self.screen_height, self.screen_width, 4))

        # 分离Alpha通道（如果需要保留Alpha，可以去掉这步）
        # 这里将ARGB转换为BGR（OpenCV默认格式）
        # 通过切片操作 [:, :, [2, 1, 0]] 实现通道交换
        bgr_array = argb_array[:, :, [2, 1, 0]]  # 交换R和B通道

        # 转换为OpenCV可用的连续数组（某些OpenCV操作需要）
        image = np.ascontiguousarray(bgr_array)
        return image

    def decode_raw_with_gzip(self, data: bytes):
        decompressed_data = gzip.decompress(data)
        image = self.decode_raw(decompressed_data)
        return image

    def capture_screenshot_raw_gzip(self):
        get_raw_gzip_cmd = (
            rf'{self.adb_path} -s {self.device_serial} exec-out "screencap | gzip -1"'
        )
        ta = time.time()
        try:
            # 获取经过gzip压缩的二进制图像数据
            screenshot_raw_gzip = subprocess.check_output(get_raw_gzip_cmd, shell=True)
            image = self.decode_raw_with_gzip(screenshot_raw_gzip)
            if image is None:
                raise RuntimeError("OpenCV failed to decode image")
        except subprocess.CalledProcessError as e:
            logger.exception("Screenshot capture failed (ADB error):", e)
            return None
        except gzip.BadGzipFile as e:
            logger.exception("Gzip decompression failed:", e)
            return None
        except Exception as e:
            logger.exception("Image processing error:", e)
            return None
        logger.debug(f"获取图片用时{time.time()-ta:.3f}s")
        return image

    def click(self, point):
        x, y = point
        x_coord = int(x * self.screen_width)
        y_coord = int(y * self.screen_height)
        logger.info(f"点击坐标: ({x_coord}, {y_coord})")
        click_cmd = f"{self.adb_path} -s {self.device_serial} shell input tap {x_coord} {y_coord}"
        subprocess.run(click_cmd, shell=True)


relative_points = [
    (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
    (0.0713, 0.8833),  # 左ALL
    (0.8281, 0.8833),  # 右礼物、自娱自乐
    (0.1640, 0.8833),  # 左礼物
    (0.4979, 0.6324),  # 本轮观望
]


"""
def operation_simple(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx == 0:  # 加入赛事
                click(relative_points[0])
                logger.info("加入赛事")
            elif idx == 1:  # 自娱自乐
                click(relative_points[2])
                logger.info("自娱自乐")
            elif idx == 2:  # 开始游戏
                click(relative_points[0])
                logger.info("开始游戏")
            elif idx in [3, 4, 5]:  # 本轮观望
                click(relative_points[4])
                logger.info("本轮观望")
            elif idx in [10, 11]:
                logger.info("下一轮")
            elif idx in [6, 7]:
                logger.info("等待战斗结束")
            elif idx == 12:  # 返回主页
                click(relative_points[0])
                logger.info("返回主页")
            break  # 匹配到第一个结果后退出


def operation(results):
    for idx, score in results:
        if score > 0.6:  # 假设匹配阈值为 0.8
            if idx in [3, 4, 5]:
                # 识别怪物类型数量，导入模型进行预测
                prediction = 0.6
                # 根据预测结果点击投资左/右
                if prediction > 0.5:
                    click(relative_points[1])  # 投资右
                    logger.info("投资右")
                else:
                    click(relative_points[0])  # 投资左
                    logger.info("投资左")
            elif idx in [1, 5]:
                click(relative_points[2])  # 点击省点饭钱
                logger.info("点击省点饭钱")
            elif idx == 2:
                click(relative_points[3])  # 点击敬请见证
                logger.info("点击敬请见证")
            elif idx in [3, 4]:
                # 保存数据
                click(relative_points[4])  # 点击下一轮
                logger.info("点击下一轮")
            elif idx == 6:
                logger.info("等待战斗结束")
            break  # 匹配到第一个结果后退出
"""


# def main():
#     while True:
#         screenshot = capture_screenshot()
#         if screenshot is not None:
#             results = match_images(screenshot, process_images)
#             results = sorted(results, key=lambda x: x[1], reverse=True)
#             print("匹配结果：", results[0])
#             operation(results)
#         time.sleep(2)


# if __name__ == "__main__":
#     main()
