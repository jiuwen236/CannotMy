import csv
import logging
import os
import subprocess
import threading
import time
import math
import cv2
import keyboard
import numpy as np
import pandas as pd
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox,
                             QGroupBox, QScrollArea, QMessageBox, QGridLayout, QSizePolicy, QGraphicsDropShadowEffect,
                             QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QColor
from sklearn.metrics.pairwise import cosine_similarity

import loadData
import recognize
import train
import auto_fetch
import similar_history_match
from train import UnitAwareTransformer
from recognize import MONSTER_COUNT, intelligent_workers_debug
from specialmonster import SpecialMonsterHandler
from predict import CannotModel

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",)
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

loadData.connect()

class ArknightsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = False
        self.game_mode = "单人"
        self.device_serial = loadData.manual_serial

        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}
        self.main_roi = None

        # 统计信息
        self.total_fill_count = 0
        self.incorrect_fill_count = 0
        self.start_time = None

        # 模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.current_prediction = 0.5
        self.current_image_name = ""

        # 添加历史对局相关属性
        self.history_visible = False
        self.history_data_loaded = True
        self.history_widget = None
        self.history_scroll_area = None

        # 初始化UI后加载历史数据
        self.history_match = similar_history_match.HistoryMatch()
        self.load_history_data()

        self.cannot_model = CannotModel() 
        # 初始化特殊怪物语言触发处理程序
        self.special_monster_handler = SpecialMonsterHandler()

        self.init_ui()
        self.load_images()

    def init_ui(self):
        self.setWindowTitle("铁鲨鱼_Arknights Neural Network")
        self.setWindowIcon(QIcon("ico/icon.ico"))
        self.setGeometry(100, 100, 1700, 800)
        self.background = QPixmap("ico/background.png")

        # TODO: 发光效果无效
        # qss = """
        #         QWidget {
        #             /* 添加发光效果 */
        #             qproperty-effect: true;
        #             qproperty-glow-color: rgba(255, 165, 0, 150);
        #             qproperty-glow-radius: 10px;
        #         }
        #         """

        # self.setStyleSheet(qss)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧面板
        left_panel = QWidget()
        left_panel.setObjectName("left_panel_id")
        left_panel.setStyleSheet("""
            QWidget#left_panel_id {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            """)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        # 人物显示区域
        monster_group = QWidget()
        monster_layout = QVBoxLayout(monster_group)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:horizontal {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollArea {
                background-color: rgba(0, 0, 0, 0);
                border:0px
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 50, 100);
                width: 12px;
                margin: 15px 0 15px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;
                border-radius: 6px;
            }
        """)

        scroll_content = QWidget()
        self.scroll_grid = QGridLayout(scroll_content)
        self.scroll_grid.setSpacing(5)
        self.scroll_grid.setContentsMargins(5, 5, 5, 5)

        # 设置5列布局
        self.COLUMNS = 7
        self.ROW_HEIGHT = 120  # 每个单元的高度

        scroll.setWidget(scroll_content)
        monster_layout.addWidget(scroll)
        left_layout.addWidget(monster_group)

        # 右侧面板 - 结果和控制区
        right_panel = QWidget()
        right_panel.setFixedWidth(550)  # 固定右侧面板宽度
        right_layout = QVBoxLayout(right_panel)

        # 顶部区域 - 输入显示
        input_display = QGroupBox()
        input_display.setStyleSheet("""
            QGroupBox {
                background-color: rgba(0, 0, 0, 120);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
                margin-top: 10px;
                padding: 10px 0;
            }
            QGroupBox::title {
                color: white;
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        input_layout = QHBoxLayout(input_display)

        # 左侧人物显示
        left_input_group = QWidget()
        left_input_layout = QHBoxLayout(left_input_group)
        self.left_input_content = QWidget()
        self.left_input_layout = QHBoxLayout(self.left_input_content)
        self.left_input_layout.setSpacing(5)
        left_input_layout.addWidget(self.left_input_content)

        # 右侧人物显示
        right_input_group = QWidget()
        right_input_layout = QHBoxLayout(right_input_group)
        self.right_input_content = QWidget()
        self.right_input_layout = QHBoxLayout(self.right_input_content)
        self.right_input_layout.setSpacing(5)
        right_input_layout.addWidget(self.right_input_content)

        # 将左右两部分添加到主输入布局
        input_layout.addWidget(left_input_group)
        input_layout.addWidget(right_input_group)

        right_layout.addWidget(input_display)

        # 中部区域 - 预测结果
        result_group = QGroupBox()
        result_group.setStyleSheet("""
            QGroupBox {
                background-color: rgba(120, 120, 120, 10);
                border-radius: 15px;
                border: 1px solid #747474;
            }
            """)
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setFont(QFont("Microsoft YaHei", 12))
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)

        result_button = QWidget()
        result_button_layout = QHBoxLayout(result_button)

        # 预测按钮 - 带样式
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setStyleSheet("""
                    QPushButton {
                        background-color: #313131;
                        color: #F3F31F;
                        border-radius: 16px;
                        padding: 8px;
                        font-weight: bold;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #414141;
                    }
                    QPushButton:pressed {
                        background-color: #212121;
                    }
                """)
        self.predict_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_entries)
        self.reset_button.setStyleSheet("""
                            QPushButton {
                                background-color: #313131;
                                color: #F3F31F;
                                border-radius: 16px;
                                padding: 8px;
                                font-weight: bold;
                                min-height: 30px;
                            }
                            QPushButton:hover {
                                background-color: #414141;
                            }
                            QPushButton:pressed {
                                background-color: #212121;
                            }
                        """)

        result_button_layout.addWidget(self.predict_button)
        result_button_layout.addWidget(self.reset_button)

        result_layout.addWidget(result_button)
        right_layout.addWidget(result_group)

        # 底部区域 - 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)

        # 第一行按钮
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)

        self.duration_label = QLabel("训练时长(小时):")
        self.duration_entry = QLineEdit("-1")
        self.duration_entry.setFixedWidth(50)

        self.auto_fetch_button = QPushButton("自动获取数据")
        self.auto_fetch_button.clicked.connect(self.toggle_auto_fetch)

        self.mode_menu = QComboBox()
        self.mode_menu.addItems(["单人", "30人"])

        self.invest_checkbox = QCheckBox("投资")

        row1_layout.addWidget(self.duration_label)
        row1_layout.addWidget(self.duration_entry)
        row1_layout.addWidget(self.auto_fetch_button)
        row1_layout.addWidget(self.mode_menu)
        row1_layout.addWidget(self.invest_checkbox)

        # 第二行按钮
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)

        self.recognize_button = QPushButton("识别")
        self.recognize_button.clicked.connect(self.recognize)

        self.fill_correct_button = QPushButton("填写√")
        self.fill_correct_button.clicked.connect(self.fill_data_correct)

        self.fill_incorrect_button = QPushButton("填写×")
        self.fill_incorrect_button.clicked.connect(self.fill_data_incorrect)

        row2_layout.addWidget(self.recognize_button)
        row2_layout.addWidget(self.fill_correct_button)
        row2_layout.addWidget(self.fill_incorrect_button)

        # 第三行按钮
        row3 = QWidget()
        row3_layout = QHBoxLayout(row3)

        self.reselect_button = QPushButton("选择范围")
        self.reselect_button.clicked.connect(self.reselect_roi)

        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QLineEdit(self.device_serial)
        self.serial_entry.setFixedWidth(60)

        self.serial_button = QPushButton("更新")
        self.serial_button.clicked.connect(self.update_device_serial)

        row3_layout.addWidget(self.reselect_button)
        row3_layout.addWidget(self.serial_label)
        row3_layout.addWidget(self.serial_entry)
        row3_layout.addWidget(self.serial_button)

        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))

        # 添加所有行到控制布局
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(row3)
        control_layout.addWidget(self.stats_label)

        right_layout.addWidget(control_group)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(main_widget)

        # 在右侧面板添加历史对局按钮
        self.history_button = QPushButton("显示历史对局")
        self.history_button.clicked.connect(self.toggle_history_panel)
        self.history_button.setStyleSheet("""
                    QPushButton {
                        background-color: #313131;
                        color: #F3F31F;
                        border-radius: 16px;
                        padding: 8px;
                        font-weight: bold;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #414141;
                    }
                    QPushButton:pressed {
                        background-color: #212121;
                    }
                """)
        right_layout.addWidget(self.history_button)

        # 连接输入框变化信号
        for entry in self.left_monsters.values():
            entry.textChanged.connect(self.update_input_display)

        # 自动获取数据的定时器
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.auto_fetch_update)

        # 连接信号
        self.mode_menu.currentTextChanged.connect(self.update_game_mode)
        self.invest_checkbox.stateChanged.connect(self.update_invest_status)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 缩放图片以适应窗口（保持宽高比）
        scaled_pixmap = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        # 居中绘制
        painter.drawPixmap(
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap
        )

    def load_images(self):
        for i in reversed(range(self.scroll_grid.count())):
            self.scroll_grid.itemAt(i).widget().setParent(None)

            # 重新计算布局
        row = 0
        col = 0

        for i in range(1, MONSTER_COUNT + 1):
            # 容器
            monster_container = QWidget()
            monster_container.setFixedHeight(self.ROW_HEIGHT)
            shadow01 = QGraphicsDropShadowEffect()
            shadow01.setBlurRadius(5)  # 模糊半径（控制发光范围）
            shadow01.setColor(QColor(0, 0, 0,120))  # 发光颜色
            shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
            monster_container.setGraphicsEffect(shadow01)

            monster_container.setStyleSheet("""
                                QWidget {
                                    border-radius: 0px;
                                }
                            """)
            container_layout = QVBoxLayout(monster_container)
            container_layout.setSpacing(2)
            container_layout.setContentsMargins(2, 2, 2, 2)

            # 人物图片
            img_label = QLabel()
            img_label.setFixedSize(60, 60)
            img_label.setAlignment(Qt.AlignCenter)

            try:
                pixmap = QPixmap(f"images/{i}.png")
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img_label.setPixmap(pixmap)
            except Exception as e:
                print(f"加载人物{i}图片错误: {str(e)}")

            # 左输入框
            left_entry = QLineEdit()
            left_entry.setFixedWidth(60)
            left_entry.setPlaceholderText("左")
            left_entry.setAlignment(Qt.AlignCenter)
            self.left_monsters[str(i)] = left_entry

            # 右输入框 (放在左输入框下方)
            right_entry = QLineEdit()
            right_entry.setFixedWidth(60)
            right_entry.setPlaceholderText("右")
            right_entry.setAlignment(Qt.AlignCenter)
            self.right_monsters[str(i)] = right_entry

            # 添加到容器
            container_layout.addWidget(img_label, 0, Qt.AlignCenter)
            container_layout.addWidget(left_entry, 0, Qt.AlignCenter)
            container_layout.addWidget(right_entry, 0, Qt.AlignCenter)

            # 添加到网格布局
            self.scroll_grid.addWidget(monster_container, row, col, Qt.AlignCenter)

            # 更新行列位置
            col += 1
            if col >= self.COLUMNS:
                col = 0
                row += 1

    def update_input_display(self):
        for i in reversed(range(self.left_input_layout.count())):
            widget = self.left_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

            # 清除右侧现有显示
        for i in reversed(range(self.right_input_layout.count())):
            widget = self.right_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

            # 检查是否有输入
        left_has_input = False
        right_has_input = False

        # 更新左侧人物显示
        for i in range(1, MONSTER_COUNT + 1):
            left_value = self.left_monsters[str(i)].text()
            right_value = self.right_monsters[str(i)].text()

            # 左侧人物显示
            if left_value.isdigit() and int(left_value) > 0:
                left_has_input = True
                monster_widget = self.create_monster_display_widget(i, left_value)
                self.left_input_layout.addWidget(monster_widget)

            # 右侧人物显示
            if right_value.isdigit() and int(right_value) > 0:
                right_has_input = True
                monster_widget = self.create_monster_display_widget(i, right_value)
                self.right_input_layout.addWidget(monster_widget)

        # 如果没有输入，显示提示
        if not left_has_input:
            self.left_input_layout.addWidget(QLabel("无"))
        if not right_has_input:
            self.right_input_layout.addWidget(QLabel("无"))

    def create_monster_display_widget(self, monster_id, count):
        """创建人物显示组件"""
        widget = QWidget()
        widget.setFixedWidth(67)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(0)  # 模糊半径（控制发光范围）
        shadow.setColor(QColor("#313131"))  # 发光颜色
        shadow.setOffset(2)  # 偏移量（0表示均匀四周发光）
        widget.setGraphicsEffect(shadow)

        widget.setStyleSheet("""
                    QWidget {
                        border-radius: 0px;
                    }
                """)

        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignCenter)

        # 人物图片
        img_label = QLabel()
        img_label.setFixedSize(70, 70)
        img_label.setAlignment(Qt.AlignCenter)

        try:
            pixmap = QPixmap(f"images/{monster_id}.png")
            if not pixmap.isNull():
                pixmap = pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
        except:
            pass

        # 数量标签
        count_label = QLabel(count)
        count_label.setAlignment(Qt.AlignCenter)
        count_label.setStyleSheet("""
            color: #EDEDED;
            font: bold 20px SimHei;
            border-radius: 5px;
            padding: 2px 5px;
            min-width: 20px;
        """)

        layout.addWidget(img_label)
        layout.addWidget(count_label)

        return widget

    def reset_entries(self):
        for entry in self.left_monsters.values():
            entry.clear()
            entry.setStyleSheet("")
        for entry in self.right_monsters.values():
            entry.clear()
            entry.setStyleSheet("")
        self.result_label.setText("预测结果将显示在这里")
        self.result_label.setStyleSheet("color: black;")
        self.update_input_display()

    def fill_data_correct(self):
        result = 'R' if self.current_prediction > 0.5 else 'L'
        self.fill_data(result)
        self.total_fill_count += 1
        self.update_statistics()

    def fill_data_incorrect(self):
        result = 'L' if self.current_prediction > 0.5 else 'R'
        self.fill_data(result)
        self.total_fill_count += 1
        self.incorrect_fill_count += 1
        self.update_statistics()

    def get_prediction(self):
        try:
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            for name, entry in self.left_monsters.items():
                value = entry.text()
                left_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            for name, entry in self.right_monsters.items():
                value = entry.text()
                right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            prediction = self.cannot_model.get_prediction(left_counts, right_counts)
            return prediction
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到模型文件，请先训练")
            return 0.5
        except RuntimeError as e:
            if "size mismatch" in str(e):
                QMessageBox.critical(self, "错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            return 0.5
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数字（0或正整数）")
            return 0.5
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时发生错误: {str(e)}")
            return 0.5

    def update_prediction(self, prediction):
        """更新预测结果显示 (单模型版本)"""
        # 模型结果处理
        right_win_prob = prediction
        left_win_prob = 1 - right_win_prob

        # 判断胜负方向
        winner = "左方" if left_win_prob > 0.5 else "右方"
        if 0.6 > left_win_prob > 0.4:
            winner = "难说"

        # 设置结果标签样式
        if winner == "左方":
            self.result_label.setStyleSheet("color: #E23F25; font: bold,14px;")
        else:
            self.result_label.setStyleSheet("color: #25ace2; font: bold,14px;")

        # 生成结果文本
        if winner != "难说":
            result_text = (f"预测胜方: {winner}\n"
                         f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n")
            
            # 添加特殊干员提示
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages
            
            # 极高概率时的特殊提示
            if left_win_prob > 0.999 or right_win_prob > 0.999:
                if left_win_prob > 0.999:
                    result_text += "\n      右边赢了我给你们口  ——克头"
                if right_win_prob > 0.999:
                    result_text += "\n      左边赢了我给你们口  ——克头"
        else:
            result_text = (f"这一把{winner}\n"
                         f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n"
                         f"难道说？难道说？难道说？\n")
            self.result_label.setStyleSheet("color: black; font: bold,24px;")
            
            # 添加特殊干员提示
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages

        self.result_label.setText(result_text)
        self.current_prediction = prediction

    def predict(self):
        prediction1 = self.get_prediction()
        self.current_prediction = prediction1
        self.update_prediction(prediction1)
        self.update_input_display()

        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()

    def recognize(self):
        if self.auto_fetch_running:
            screenshot = loadData.capture_screenshot()
        else:
            screenshot = None

        if self.no_region:
            if self.first_recognize:
                self.main_roi = [
                    (int(0.2479 * loadData.screen_width), int(0.8410 * loadData.screen_height)),
                    (int(0.7526 * loadData.screen_width), int(0.9510 * loadData.screen_height))
                ]
                adb_path = loadData.adb_path
                device_serial = loadData.device_serial
                subprocess.run(f'{adb_path} connect {device_serial}', shell=True, check=True)
                self.first_recognize = False
            screenshot = loadData.capture_screenshot()

        results = recognize.process_regions(self.main_roi, screenshot=screenshot)
        self.reset_entries()

        for res in results:
            if 'error' not in res:
                region_id = res['region_id']
                matched_id = res['matched_id']
                number = res['number']
                if matched_id != 0:
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.setText(str(number))
                    if entry.text():
                        entry.setStyleSheet("background-color: yellow;")

        prediction1 = self.get_prediction()
        self.current_prediction = prediction1
        self.update_prediction(prediction1)

        self.update_input_display()
        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()

    def load_history_data(self):
        """加载历史对局数据"""
        self.past_left = self.history_match.past_left
        self.past_right = self.history_match.past_right
        self.labels = self.history_match.labels
        # 组合特征
        self.feat_past = self.history_match.feat_past
        self.N_history = self.history_match.N_history

    def toggle_history_panel(self):
        """切换历史对局面板的显示"""
        if not self.history_visible:
            self.show_history_panel()
            self.history_button.setText("隐藏历史对局")
        else:
            self.hide_history_panel()
            self.history_button.setText("显示历史对局")
        self.history_visible = not self.history_visible

    def show_history_panel(self):
        """显示历史对局面板"""
        if not self.history_data_loaded:
            QMessageBox.warning(self, "警告", "历史数据加载失败，无法显示历史对局")
            return

        # 创建滚动区域
        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setFixedWidth(540)
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setStyleSheet("""
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:horizontal {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollArea {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 50, 100);
                width: 12px;
                margin: 15px 0 15px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;
                border-radius: 6px;
            }
        """)

        # 创建内容部件
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignTop)

        # 渲染历史对局
        self.render_similar_matches()

        # 设置滚动区域内容
        self.history_scroll_area.setWidget(self.history_widget)

        # 添加到主界面
        self.centralWidget().layout().addWidget(self.history_scroll_area)

    def hide_history_panel(self):
        """隐藏历史对局面板"""
        if self.history_scroll_area:
            self.history_scroll_area.setParent(None)
            self.history_scroll_area = None
            self.history_widget = None

    def render_similar_matches(self):
        try:
            # 获取当前输入
            cur_left = np.zeros(56, dtype=float)
            cur_right = np.zeros(56, dtype=float)
            for name, entry in self.left_monsters.items():
                v = entry.text()
                if v.isdigit():
                    cur_left[int(name) - 1] = float(v)
            for name, entry in self.right_monsters.items():
                v = entry.text()
                if v.isdigit():
                    cur_right[int(name) - 1] = float(v)

            self.history_match.render_similar_matches(cur_left, cur_right)
            sims = self.history_match.sims
            top_indices = self.history_match.top20_idx

            # 清空现有内容
            for i in reversed(range(self.history_layout.count())):
                self.history_layout.itemAt(i).widget().setParent(None)

            # 添加标题
            title_label = QLabel(f"错题本")
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(0)  # 模糊半径（控制发光范围）
            shadow.setColor(QColor("#313131"))  # 发光颜色
            shadow.setOffset(2)  # 偏移量（0表示均匀四周发光）
            title_label.setGraphicsEffect(shadow)

            title_label.setStyleSheet("""
                                QWidget {
                                    border-radius: 0px;
                                    font-size: 24px;
                                    font-weight: bold;
                                    color: white;
                                }
                            """)
            self.history_layout.addWidget(title_label)

            # 渲染每个历史对局
            for idx in top_indices:
                self.add_history_match(idx, sims[idx])

        except Exception as e:
            print(f"渲染历史对局失败: {str(e)}")

    def add_history_match(self, idx, similarity):
        """添加单个历史对局到面板"""
        # 获取历史数据
        left = self.past_left[idx]
        right = self.past_right[idx]
        result = self.labels[idx]

        # 获取当前对局的左右单位
        cur_left = np.zeros(56, dtype=float)
        cur_right = np.zeros(56, dtype=float)
        for name, entry in self.left_monsters.items():
            v = entry.text()
            if v.isdigit(): cur_left[int(name)-1] = float(v)
        for name, entry in self.right_monsters.items():
            v = entry.text()
            if v.isdigit(): cur_right[int(name)-1] = float(v)

        # 计算当前对局和历史对局的相似度(不镜像和镜像两种情况)
        setL_cur = set(np.where(cur_left > 0)[0])
        setR_cur = set(np.where(cur_right > 0)[0])
        setL_past = set(np.where(left > 0)[0])
        setR_past = set(np.where(right > 0)[0])
        
        # 判断是否需要镜像历史对局
        should_swap = len(setL_cur ^ setR_past) + len(setR_cur ^ setL_past) < \
                     len(setL_cur ^ setL_past) + len(setR_cur ^ setR_past)

        # 创建对局容器
        match_widget = QWidget()
        match_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 150);
                border-radius: 10px;
                padding: 0px;
                margin: 5px;
            }
        """)
        match_widget.setFixedSize(500, 150)
        match_layout = QVBoxLayout(match_widget)

        # 添加左右阵容
        teams_widget = QWidget()
        teams_layout = QHBoxLayout(teams_widget)

        # 根据是否需要镜像决定显示方向
        if should_swap:
            left_team = self.create_team_widget("右方", right, result == 'R')
            right_team = self.create_team_widget("左方", left, result == 'L')
        else:
            left_team = self.create_team_widget("左方", left, result == 'L')
            right_team = self.create_team_widget("右方", right, result == 'R')

        teams_layout.addWidget(left_team)
        teams_layout.addWidget(right_team)
        match_layout.addWidget(teams_widget)

        self.history_layout.addWidget(match_widget)

    def create_team_widget(self, side, counts, is_winner):
        """创建单个队伍显示部件"""
        team_widget = QWidget()
        team_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {'rgba(250, 250, 50, 150)' if is_winner else 'rgba(50, 50, 50, 100)'};
                border-radius: 8px;
                padding: 0px;
                margin: 0px;
            }}
        """)

        layout = QVBoxLayout(team_widget)

        # 显示区域
        ops_widget = QWidget()
        shadow01 = QGraphicsDropShadowEffect()
        shadow01.setBlurRadius(5)  # 模糊半径（控制发光范围）
        shadow01.setColor(QColor(0, 0, 0, 120))  # 发光颜色
        shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
        ops_widget.setGraphicsEffect(shadow01)

        ops_widget.setStyleSheet("""
                                        QWidget {
                                            background-color: rgba(0, 0, 0, 0);
                                            border-radius: 0px;
                                            padding: 0px;
                                            margin: 0px;
                                        }
                                    """)
        ops_layout = QHBoxLayout(ops_widget)
        ops_layout.setSpacing(5)
        ops_layout.setContentsMargins(0, 0, 0, 0)

        for i, count in enumerate(counts):
            if count > 0:
                # 创建干员显示
                op_widget = QWidget()
                op_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0); padding: 0px 0;margin: 0px;")
                op_layout = QVBoxLayout(op_widget)
                op_layout.setContentsMargins(0, 0, 0, 0)
                op_layout.setAlignment(Qt.AlignCenter)

                # 干员图片
                img_label = QLabel()
                img_label.setFixedSize(60, 60)
                img_label.setAlignment(Qt.AlignCenter)
                try:
                    pixmap = QPixmap(f"images/{i + 1}.png")
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        img_label.setPixmap(pixmap)
                except:
                    pass

                # 数量标签
                count_label = QLabel(str(int(count)))
                count_label.setAlignment(Qt.AlignCenter)
                count_label.setStyleSheet("""
                            color: #EDEDED;
                            font: bold 20px SimHei;
                            min-width: 20px;
                        """)

                op_layout.addWidget(img_label, stretch=3)
                op_layout.addWidget(count_label, stretch=1)
                ops_layout.addWidget(op_widget)

        layout.addWidget(ops_widget)
        return team_widget

    def reselect_roi(self):
        self.main_roi = recognize.select_roi()
        self.no_region = False

    # def calculate_average_yellow(self, image):
    #     if image is None:
    #         print("图像加载失败")
    #         return None
    #     height, width, _ = image.shape
    #     point_color = image[0, 0]
    #     blue, green, red = point_color
    #     is_yellow = (red > 150 and green > 150 and blue < 100)
    #     return is_yellow

    # def save_statistics_to_log(self):
    #     elapsed_time = time.time() - self.start_time if self.start_time else 0
    #     hours, remainder = divmod(elapsed_time, 3600)
    #     minutes, _ = divmod(remainder, 60)
    #     stats_text = (f"总共填写次数: {self.total_fill_count}\n"
    #                   f"填写×次数: {self.incorrect_fill_count}\n"
    #                   f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n")
    #     with open("log.txt", "a") as log_file:
    #         log_file.write(stats_text)

    def toggle_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch = auto_fetch.AutoFetch(
                self.game_mode,
                self.is_invest,
                reset=self.reset_entries,
                recognizer=self.recognize,
                updater=self.update_statistics,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
                training_duration=float(self.duration_entry.text()) * 3600,  # 获取训练时长
            )
            self.auto_fetch.start_auto_fetch()
            return
            self.auto_fetch_running = True
            self.auto_fetch_button.setText("停止自动获取")
            self.start_time = time.time()
            self.total_fill_count = 0
            self.incorrect_fill_count = 0
            self.update_statistics()
            self.training_duration = float(self.duration_entry.text()) * 3600
            self.timer.start(500)  # 每500ms检查一次
        else:
            self.auto_fetch.stop_auto_fetch()
            return
            self.auto_fetch_running = False
            self.auto_fetch_button.setText("自动获取数据")
            self.update_statistics()
            self.save_statistics_to_log()
            self.timer.stop()

    # def auto_fetch_update(self):
    #     if not hasattr(self, 'auto_fetch') or not self.auto_fetch.auto_fetch_running:
    #         self.timer.stop()
    #         return

    #     try:
    #         # 使用AutoFetch类的auto_fetch_data方法
    #         self.auto_fetch.auto_fetch_data()
            
    #         # 更新统计信息
    #         self.update_statistics()

    #         # 检查训练时长是否结束
    #         elapsed_time = time.time() - self.auto_fetch.start_time
    #         if self.auto_fetch.training_duration != -1 and elapsed_time >= self.auto_fetch.training_duration:
    #             self.auto_fetch.stop_auto_fetch()
    #             self.auto_fetch_button.setText("自动获取数据")
    #             self.save_statistics_to_log()
    #             self.timer.stop()

    #         # ESC键停止自动获取
    #         if keyboard.is_pressed('esc'):
    #             self.auto_fetch.stop_auto_fetch()
    #             self.auto_fetch_button.setText("自动获取数据")
    #             self.save_statistics_to_log()
    #             self.timer.stop()

    #     except Exception as e:
    #         print(f"自动获取数据出错: {str(e)}")
    #         if hasattr(self, 'auto_fetch'):
    #             self.auto_fetch.stop_auto_fetch()
    #         self.auto_fetch_button.setText("自动获取数据")
    #         self.save_statistics_to_log()
    #         self.timer.stop()

    def update_statistics(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (f"总共填写次数: {self.total_fill_count},    "
                      f"填写×次数: {self.incorrect_fill_count},    "
                      f"当次运行时长: {int(hours)}小时{int(minutes)}分钟")
        self.stats_label.setText(stats_text)

    def update_device_serial(self):
        new_serial = self.serial_entry.text()
        loadData.set_device_serial(new_serial)
        loadData.device_serial = None
        loadData.get_device_serial()
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {new_serial}")

    def start_callback(self):
        self.auto_fetch_button.setText("停止自动获取数据")

    def stop_callback(self):
        self.auto_fetch_button.setText("自动获取数据")

    def update_game_mode(self, mode):
        self.game_mode = mode

    def update_invest_status(self, state):
        self.is_invest = state == Qt.Checked

    def update_result(self, text):
        self.result_label.setText(text)

    def update_stats(self, total, incorrect, duration):
        stats_text = f"总共: {total}, 错误: {incorrect}, 时长: {duration}"
        self.stats_label.setText(stats_text)

    def update_image_display(self, qimage):
        self.image_display.setPixmap(QPixmap.fromImage(qimage).scaled(
            self.image_display.width(),
            self.image_display.height(),
            Qt.KeepAspectRatio
        ))


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec_()