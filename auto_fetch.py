import csv
import os
import time
from tkinter import image_names
from xmlrpc.client import Boolean

import cv2
import numpy as np
from sympy import N
import loadData
from recognize import MONSTER_COUNT, intelligent_workers_debug


class AutoFetch:
    def __init__(self, game_mode, is_invest, reset, recognizer, updater):
        self.game_mode = game_mode  # 游戏模式（30人或自娱自乐）
        self.is_invest = is_invest  # 是否投资
        self.current_prediction = 0.5  # 当前预测结果，初始值为0.5
        self.incorrect_fill_count = 0  # 填写错误次数
        self.total_fill_count = 0  # 总填写次数
        self.reset = reset  # 重置填写数据的函数
        self.recognizer = recognizer  # 识别怪物类型数量的函数
        self.updater = updater  # 更新统计信息的函数
        self.image = None
        self.image_name = ""

    @staticmethod
    def fill_data(result, left_monsters, right_monsters, image, image_name):
        image_data = np.zeros((1, MONSTER_COUNT * 2))
        for name, entry in left_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) - 1] = int(value)
        for name, entry in right_monsters.items():
            value = entry.get()
            if value.isdigit():
                image_data[0][int(name) + MONSTER_COUNT - 1] = int(value)
        image_data = np.append(image_data, result)
        image_data = np.nan_to_num(image_data, nan=-1)  # 替换所有NaN为-1

        # 将数据转换为列表，并添加图片名称
        data_row = image_data.tolist()
        if intelligent_workers_debug:  # 如果处于debug模式
            data_row.append(image_name)
            # ==================在这里保存人工审核图片到本地==================
            if image is not None:
                os.makedirs("data/images", exist_ok=True)
                image_path = os.path.join("data/images", image_name)
                cv2.imwrite(image_path, image)

        with open("arknights.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

    @staticmethod
    def calculate_average_yellow(image):
        # 检测左上角一点是否为黄色
        if image is None:
            print(f"图像加载失败")
            return None
        height, width, _ = image.shape
        # 取左上角(0,0)点
        point_color = image[0, 0]
        # 提取BGR通道值
        blue, green, red = point_color
        # 判断是否为黄色 (黄色RGB值大致为R高、G高、B低)
        is_yellow: Boolean = red > 150 and green > 150 and blue < 100
        return is_yellow

    @staticmethod
    def save_recoginze_image(results, screenshot):
        """
        生成复核图片
        """
        x1 = int(0.2479 * loadData.screen_width)
        y1 = int(0.8444 * loadData.screen_height)
        x2 = int(0.7526 * loadData.screen_width)
        y2 = int(0.9491 * loadData.screen_height)
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
        if screenshot is not None:
            # 创建images目录（如果不存在）
            os.makedirs("data/images", exist_ok=True)
        # 将处理的怪物 ID 拼接到文件名中
        monster_ids_str = "_".join(map(str, processed_monster_ids))
        current_image_name = f"{timestamp}_{monster_ids_str}.png"
        current_image = cv2.resize(
            roi, (roi.shape[1] // 2, roi.shape[0] // 2)
        )  # 保存缩放后的图片到内存
        return current_image, current_image_name

    def auto_fetch_data(self, left_monsters, right_monsters):
        relative_points = [
            (0.9297, 0.8833),  # 右ALL、返回主页、加入赛事、开始游戏
            (0.0713, 0.8833),  # 左ALL
            (0.8281, 0.8833),  # 右礼物、自娱自乐
            (0.1640, 0.8833),  # 左礼物
            (0.4979, 0.6324),  # 本轮观望
        ]
        screenshot = loadData.capture_screenshot()
        if screenshot is not None:
            results = loadData.match_images(screenshot, loadData.process_images)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            # print("匹配结果：", results[0])
            for idx, score in results:
                if score > 0.5:
                    if idx == 0:
                        loadData.click(relative_points[0])
                        print("加入赛事")
                    elif idx == 1:
                        if self.game_mode == "30人":
                            loadData.click(relative_points[1])
                            print("竞猜对决30人")
                            time.sleep(2)
                            loadData.click(relative_points[0])
                            print("开始游戏")
                        else:
                            loadData.click(relative_points[2])
                            print("自娱自乐")
                    elif idx == 2:
                        loadData.click(relative_points[0])
                        print("开始游戏")
                    elif idx in [3, 4, 5, 15]:
                        time.sleep(1)
                        # 归零
                        self.reset()
                        # 识别怪物类型数量
                        self.current_prediction, recognize_results, screenshot = self.recognizer()
                        # 人工审核保存测试用例截图
                        if intelligent_workers_debug:  # 如果处于debug模式且处于自动模式
                            self.image, self.image_name = self.save_recoginze_image(
                                recognize_results, screenshot
                            )
                        # 点击下一轮
                        if self.is_invest:  # 投资
                            # 根据预测结果点击投资左/右
                            if self.current_prediction > 0.5:
                                if idx == 4:
                                    loadData.click(relative_points[0])
                                else:
                                    loadData.click(relative_points[2])
                                print("投资右")
                            else:
                                if idx == 4:
                                    loadData.click(relative_points[1])
                                else:
                                    loadData.click(relative_points[3])
                                print("投资左")
                            if self.game_mode == "30人":
                                time.sleep(20)  # 30人模式下，投资后需要等待20秒
                        else:  # 不投资
                            loadData.click(relative_points[4])
                            print("本轮观望")
                            time.sleep(5)

                    elif idx in [8, 9, 10, 11]:
                        # 判断本次是否填写错误
                        if self.calculate_average_yellow(screenshot):
                            self.fill_data(
                                "L", left_monsters, right_monsters, self.image, self.image_name
                            )
                            if self.current_prediction > 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据左赢")
                        else:
                            self.fill_data(
                                "R", left_monsters, right_monsters, self.image, self.image_name
                            )
                            if self.current_prediction < 0.5:
                                self.incorrect_fill_count += 1  # 更新填写×次数
                            print("填写数据右赢")
                        self.total_fill_count += 1  # 更新总填写次数
                        self.updater()  # 更新统计信息
                        print("下一轮")
                        # 为填写数据操作设置冷却期
                        time.sleep(10)
                    elif idx in [6, 7, 14]:
                        print("等待战斗结束")
                    elif idx in [12, 13]:  # 返回主页
                        loadData.click(relative_points[0])
                        print("返回主页")
                    break  # 匹配到第一个结果后退出
