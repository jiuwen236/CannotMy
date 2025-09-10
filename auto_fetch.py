import csv
import datetime
from enum import Enum, auto
import logging
from pathlib import Path
import threading
import time
from typing import Literal
import cv2
import numpy as np
from sympy import N
import loadData
from recognize import MONSTER_COUNT, intelligent_workers_debug, RecognizeMonster
from predict import CannotModel
from collections.abc import Callable
from collections import deque

FINAL_SCREENSHOT = False  # 是否保存最终结果截图

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

process_images = [cv2.imread(f"images/process/{i}.png") for i in range(17)]  # 16个模板
# 预处理模板：提前截取底部1/4区域，避免运行时重复计算
process_images_quarter = [template[int(template.shape[0] * 3 / 4) :, :] if template is not None else None for template in process_images]

def match_images(screenshot, templates, exclude_indices):
    screenshot = cv2.resize(screenshot, (1920, 1080))
    # 只截取一次屏幕底部1/4区域
    screenshot_quarter = screenshot[int(screenshot.shape[0] * 3 / 4) :, :]
    results = []
    for idx, template_quarter in enumerate(process_images_quarter):
        if idx in exclude_indices or template_quarter is None:
            continue
        res = cv2.matchTemplate(screenshot_quarter, template_quarter, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        results.append((idx, max_val))
        # 如果找到高置信度匹配(>0.9)且不在排除列表中，可以提前退出
        if max_val > 0.9:
            break
    return results

class AutoFetch:
    def __init__(
        self,
        adb_connector: loadData.AdbConnector,
        game_mode,
        is_invest_callback,  # 改为回调函数
        update_prediction_callback: Callable[[float], None],
        update_monster_callback: Callable[[list], None],
        updater: Callable[[], None],
        start_callback: Callable[[], None],
        stop_callback: Callable[[], None],
        training_duration,
    ):
        self.adb_connector = adb_connector
        self.game_mode = game_mode  # 游戏模式（30人或自娱自乐）
        self.is_invest_callback = is_invest_callback  # 投资状态回调函数
        self.current_prediction = 0.5  # 当前预测结果，初始值为0.5
        self.recognize_results = []  # 识别结果列表
        self.incorrect_fill_count = 0  # 填写错误次数
        self.total_fill_count = 0  # 总填写次数
        self.update_prediction_callback = update_prediction_callback
        self.update_monster_callback = update_monster_callback
        self.updater = updater  # 更新统计信息的函数
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.image = None  # 当前图片
        self.image_name = ""  # 当前图片名称
        self.auto_fetch_running = False  # 自动获取数据的状态
        self.start_time = time.time()  # 记录开始时间
        self.training_duration = training_duration  # 训练时长
        self.data_folder = Path(f"data")  # 数据文件夹路径
        self.image_buffer = deque(maxlen=5)  # 图片缓存队列，设置队列长短来保存结算前的图片
        self.recognizer = RecognizeMonster()
        self.cannot_model = CannotModel()
        # self.field_recognizer = FieldRecognizer()  # 场地识别器
        self.last_idx = 0

    def fill_data(self, battle_result, recoginze_results, image, image_name, result_image):
        # 获取队列头的图片
        if self.image_buffer:
            _, previous_image, _ = self.image_buffer[0]  # 获取队列头的图片
        else:
            logger.error("图片缓存队列为空，无法获取图片")
            previous_image = None

        if previous_image is None:
            logger.error("未找到1秒前的图片，无法保存")
            return
        image_data = np.zeros((1, MONSTER_COUNT * 2))

        for res in recoginze_results:
            region_id = res["region_id"]
            if "error" not in res:
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:  # 左侧怪物
                        image_data[0][matched_id - 1] = number
                    else:  # 右侧怪物
                        image_data[0][matched_id + MONSTER_COUNT - 1] = number
            else:
                logger.error(f"存在错误，本次不填写")
                return

        image_data = np.append(image_data, battle_result)

        if not np.any(image_data):
            logger.error("数据全为零，跳过保存")
            return

        image_data = np.nan_to_num(image_data, nan=-1)  # 替换所有NaN为-1

        # 将数据转换为列表，并添加图片名称
        data_row = image_data.tolist()
        # 保存数据
        start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(
            r"%Y_%m_%d__%H_%M_%S"
        )

        if intelligent_workers_debug:  # 如果处于debug模式，保存人工审核图片到本地
            image_name = image_name + ".jpg"
            data_row.append(image_name)
            if image is not None:
                image_path = self.data_folder / "images" / image_name
                # 优先使用 cv2.imwrite（但在 Windows+中文路径下可能返回 False）
                saved = False
                try:
                    saved = cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                except Exception as e:
                    logger.debug(f"cv2.imwrite 抛出异常: {e}")
                if saved:
                    logger.info(f"保存图片到 {image_path}")
                else:
                    # 回退：使用 cv2.imencode 编码后以二进制写入文件（可处理 unicode 路径问题）
                    try:
                        ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            with open(str(image_path), 'wb') as f:
                                f.write(buf.tobytes())
                            logger.info(f"通过 imencode 回退，保存图片到 {image_path}")
                        else:
                            logger.error(f"cv2.imencode 返回失败，无法保存图片: {image_path}")
                    except Exception as e:
                        logger.exception(f"使用回退方法保存图片失败: {image_path} -> {e}")

            # if previous_image is not None:
            #     image_path = self.data_folder / "images" / (image_name+"1s.png")
            #     cv2.imwrite(image_path, previous_image)
            #     logger.info(f"保存1秒前的图片到 {image_path}")

            # 新增保存结果图片逻辑
            # if self.image_name:
            #     result_image_name = self.image_name.replace(".png", "_result.png")
            #     # 缩放到128像素高度
            #     (h, w) = result_image.shape[:2]
            #     new_height = 128
            #     resized_image = cv2.resize(result_image, (int(w * (new_height / h)), new_height))
            #     image_path = self.data_folder / "images" / result_image_name
            #     cv2.imwrite(str(image_path), resized_image)
            #     logger.info(f"保存结果图片到 {image_path}")
        with open(self.data_folder / "arknights.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_row)
        logger.info(f"写入csv完成")

    @staticmethod
    def calculate_average_yellow(image):
        def get_saturation(bgr):
            # 将BGR转换为0-1范围后计算饱和度
            b, g, r = [x / 255.0 for x in bgr]
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            return (delta / cmax) * 255 if cmax != 0 else 0  # 返回0-255范围的饱和度值

        if image is None:
            logger.error("图像加载失败")
            return None

        height, width, _ = image.shape

        # 获取左上角和右上角颜色
        left_top = image[0, 0]
        right_top = image[0, width - 1]  # 右上角坐标为(width-1, 0)

        # 计算饱和度
        sat_left = get_saturation(left_top)
        sat_right = get_saturation(right_top)

        # 计算饱和度差值
        saturation_diff = sat_left - sat_right

        # 检查差值是否符合要求，平局或者其他两边相等会被这个筛选掉
        if abs(saturation_diff) <= 20:
            logger.error(f"饱和度差值不足20 (左:{sat_left:.1f} vs 右:{sat_right:.1f})")
            return None

        # 返回左上角是否比右上角饱和度更高
        return saturation_diff > 20

    def save_recoginze_image(self, results, screenshot):
        """
        生成复核图片
        """
        x1 = int(0.2479 * self.adb_connector.screen_width)
        y1 = int(0.8444 * self.adb_connector.screen_height)
        x2 = int(0.7526 * self.adb_connector.screen_width)
        y2 = int(0.9491 * self.adb_connector.screen_height)
        # 截取指定区域
        roi = screenshot[y1:y2, x1:x2]
        # 处理结果
        processed_monster_ids = []  # 用于存储处理的怪物 ID
        for res in results:
            if "error" not in res:
                matched_id = res["matched_id"]
                if matched_id != 0:
                    processed_monster_ids.append(matched_id)  # 记录处理的怪物 ID
        # 生成唯一的文件名（使用时间戳）
        timestamp = int(time.time())
        # 将处理的怪物 ID 拼接到文件名中
        monster_ids_str = "_".join(map(str, processed_monster_ids))
        current_image_name = f"{timestamp}_{monster_ids_str}"
        current_image = cv2.resize(
            roi, (roi.shape[1] // 2, roi.shape[0] // 2)
        )  # 保存缩放后的图片到内存
        return current_image, current_image_name

    def save_statistics_to_log(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.total_fill_count}\n"
            f"填写×次数: {self.incorrect_fill_count}\n"
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n"
        )
        with open("log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(stats_text)

    def recognize_and_predict(self, screenshot = None):
        if screenshot is None:
            screenshot = self.adb_connector.capture_screenshot()
        self.recognize_results = self.recognizer.process_regions(screenshot)
        # 获取预测结果
        self.update_monster_callback(self.recognize_results)
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        try:
            for res in self.recognize_results:
                region_id = res['region_id']
                matched_id = res['matched_id']
                number = res['number']
                if matched_id == 0:
                    continue
                if region_id < 3:
                    left_counts[matched_id -1] = number
                else:
                    right_counts[matched_id -1] = number
                if 'error' in res:
                    logger.error("识别结果有错误，本轮跳过")
        except Exception as e:
            logger.exception(f"处理识别结果时出错: {e}")
            return
        #收集数据阶段无模型，不进行结果预测
        self.current_prediction = self.cannot_model.get_prediction(left_counts, right_counts)
        self.update_prediction_callback(self.current_prediction)

        # 人工审核保存测试用截图
        if intelligent_workers_debug:  # 如果处于debug模式且处于自动模式
            self.image, self.image_name = self.save_recoginze_image(
                self.recognize_results, screenshot
            )
            # ==============暂时保存图片全部================
            self.image=screenshot  # 注释掉这行，避免覆盖处理过的图片

    def battle_result(self, screenshot):
        # 判断本次是否填写错误，结果不等于None（不是平局或者其他）才能继续
        if self.calculate_average_yellow(screenshot) != None:
            if self.calculate_average_yellow(screenshot):
                self.fill_data(
                    "L", self.recognize_results, self.image, self.image_name, screenshot
                )
                if self.current_prediction > 0.5:
                    self.incorrect_fill_count += 1  # 更新填写×次数
                logger.info("填写数据左赢")
            else:
                self.fill_data(
                    "R", self.recognize_results, self.image, self.image_name, screenshot
                )
                if self.current_prediction < 0.5:
                    self.incorrect_fill_count += 1  # 更新填写×次数
                logger.info("填写数据右赢")
            self.total_fill_count += 1  # 更新总填写次数
            self.updater()  # 更新统计信息
            logger.info("======下一轮======")
            # 为填写数据操作设置冷却期
            # 平局或者其他也照常休息5秒

    def auto_fetch_data(self):
        relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        eight_people = False  # 是否8人模式
        timea = time.time()
        screenshot = self.adb_connector.capture_screenshot()
        if screenshot is None:
            logger.error("截图失败，无法继续操作")
            time.sleep(1)
            return

        # 保存当前截图及其信息到缓冲区
        timestamp = int(time.time())
        self.image_buffer.append((timestamp, screenshot.copy(), []))

        # 减少需要的比对数，加快处理速度
        exclude_images = []
        images_not_30 = [2, 4, 5, 7, 12]
        images_not_single = [13, 14, 16]
        images_invest = [8, 9]
        images_not_wait_res = [0, 1, 2]
        invest_status = self.is_invest_callback()  # 获取投资状态
        if self.game_mode == "30人":
            exclude_images += images_not_30
        else:
            exclude_images += images_not_single
            if not invest_status:
                exclude_images += images_invest
        if self.last_idx in [6, 7, 14]:  # 等待战斗结束
            exclude_images += images_not_wait_res

        results = match_images(screenshot, None, exclude_images)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        logger.debug(f"处理图片总用时：{time.time()-timea:.3f}s")
        # logger.info("匹配结果：", results[0])
        for idx, score in results:
            if score > 0.5:
                if idx == 0:
                    self.adb_connector.click(relative_points[0])
                    logger.info("加入赛事")
                elif idx == 1:
                    if eight_people:
                        self.adb_connector.click((0.5, 0.8333))
                        self.adb_connector.click(relative_points[0])
                        logger.info("8人模式")
                    elif self.game_mode == "30人":
                        self.adb_connector.click(relative_points[1])
                        logger.info("竞猜对决30人")
                        time.sleep(2)
                        self.adb_connector.click(relative_points[0])
                        logger.info("开始游戏")
                    else:
                        self.adb_connector.click(relative_points[2])
                        logger.info("自娱自乐")
                elif idx == 2:
                    self.adb_connector.click(relative_points[0])
                    logger.info("开始游戏")
                elif idx in [3, 4, 5, 15]:
                    time.sleep(1)
                    # 识别怪物类型数量
                    screenshot = self.adb_connector.capture_screenshot()
                    self.recognize_and_predict(screenshot)
                    # 点击下一轮
                    if invest_status:  # 投资
                        # 根据预测结果点击投资左/右
                        if self.current_prediction > 0.5:
                            # 点两次防止失灵
                            self.adb_connector.click(relative_points[0])
                            if not eight_people:
                                self.adb_connector.click(relative_points[2])
                            logger.info("投资右")
                        else:
                            self.adb_connector.click(relative_points[1])
                            if not eight_people:
                                self.adb_connector.click(relative_points[3])
                            logger.info("投资左")
                    else:  # 不投资
                        self.adb_connector.click(relative_points[4])
                        logger.info("本轮观望")
                    time.sleep(3)
                    # 30人模式下，投资后需要等待20秒
                    if self.game_mode == "30人" or eight_people:
                        sleep_time = max(22 - (time.time() - timea), 0)  
                        time.sleep(sleep_time)

                elif idx in [8, 9, 10, 11]:
                    self.battle_result(screenshot)
                    time.sleep(5)
                    if self.game_mode == "30人" or eight_people:
                        time.sleep(5)
                elif idx in [6, 7, 14]:
                    logger.info("等待战斗结束")
                elif idx in [12, 13, 16]:  # 返回主页
                    # 新增一个截图方法，记录最终结果
                    if FINAL_SCREENSHOT and self.game_mode == "30人":
                        if self.last_idx not in [12, 13, 16]:  # 仅在首次进入等待状态时保存一次
                            time.sleep(2)
                            screenshot = self.adb_connector.capture_screenshot()
                            timestamp_str = datetime.datetime.fromtimestamp(time.time()).strftime(
                                "%Y_%m_%d__%H_%M_%S"
                            )
                            final_name = f"{timestamp_str}.jpg"
                            final_path = self.data_folder / "final_results" / final_name
                            saved = False
                            try:
                                saved = cv2.imwrite(str(final_path), screenshot, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            except Exception as e:
                                logger.debug(f"cv2.imwrite 抛出异常(最终结果): {e}")
                            if not saved:
                                try:
                                    ok, buf = cv2.imencode(".jpg", screenshot, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                    if ok:
                                        with open(str(final_path), "wb") as f:
                                            f.write(buf.tobytes())
                                        logger.info(f"通过 imencode 回退，保存最终结果图片到 {final_path}")
                                    else:
                                        logger.error(f"cv2.imencode 返回失败，无法保存最终结果图片: {final_path}")
                                except Exception as e:
                                    logger.exception(f"最终结果图片保存失败: {final_path} -> {e}")
                            else:
                                logger.info(f"保存最终结果图片到 {final_path}")
                    self.adb_connector.click(relative_points[0])
                    logger.info("返回主页")
                self.last_idx = idx
                break  # 匹配到第一个结果后退出
        sleep_time = max(1 - (time.time() - timea), 0)  # 每次循环至少1秒
        time.sleep(sleep_time)

    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            try:
                self.auto_fetch_data()
                elapsed_time = time.time() - self.start_time
                if self.training_duration != -1 and elapsed_time >= self.training_duration:
                    logger.info("已达到设定时长，结束自动获取")
                    break
            except Exception as e:
                logger.exception(f"自动获取数据出错:\n{e}")
                break
        else:
            logger.info("auto_fetch_running is False, exiting loop")
            return
        # 不通过按钮结束自动获取
        logger.info("break auto_fetch_loop")
        self.stop_auto_fetch()

    def start_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch_running = True
            self.start_time = time.time()
            start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(
                r"%Y_%m_%d__%H_%M_%S"
            )
            pic_str = "pic" if intelligent_workers_debug else "no_pic"
            self.data_folder = Path(f"data/{self.game_mode}_{pic_str}_{start_time}")
            logger.info(f"创建文件夹: {self.data_folder}")
            self.data_folder.mkdir(parents=True, exist_ok=True)  # 创建文件夹
            (self.data_folder / "images").mkdir(parents=True, exist_ok=True)
            if FINAL_SCREENSHOT and self.game_mode == "30人":
                (self.data_folder / "final_results").mkdir(parents=True, exist_ok=True)  # 新增最终结果目录
            with open(self.data_folder / "arknights.csv", "w", newline="") as file:
                header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
                header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
                header += ["Result", "ImgPath"]
                writer = csv.writer(file)
                writer.writerow(header)
            self.log_file_handler = logging.FileHandler(
                self.data_folder / f"AutoFetch_{start_time}.log", "a", "utf-8"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
            )
            self.log_file_handler.setFormatter(file_formatter)
            self.log_file_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(self.log_file_handler)
            threading.Thread(target=self.auto_fetch_loop).start()
            logger.info("自动获取数据已启动")
            self.start_callback()
        else:
            logger.warning("自动获取数据已在运行中，请勿重复启动。")

    def stop_auto_fetch(self):
        self.auto_fetch_running = False
        self.save_statistics_to_log()
        logger.info("停止自动获取")
        self.stop_callback()
        logging.getLogger().removeHandler(self.log_file_handler)
        # 结束自动获取数据的线程
