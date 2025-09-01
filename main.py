import json
import logging

import subprocess
import sys
import time
import toml
import numpy as np
from pathlib import Path
import onnxruntime # workaround: Pre-import to avoid ImportError: DLL load failed while importing onnxruntime_pybind11_state: 动态链接库(DLL)初始化例程失败。
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox,
                             QGroupBox, QScrollArea, QMessageBox, QGridLayout, QSizePolicy, QGraphicsDropShadowEffect,
                             QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QColor
import PyQt6.QtCore as QtCore

import loadData
import auto_fetch
import similar_history_match
import recognize
from recognize import MONSTER_COUNT
from specialmonster import SpecialMonsterHandler
import data_package
import winrt_capture

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from predict import CannotModel
    from train import UnitAwareTransformer
    logger.info("Using PyTorch model for predictions.")
except:
    from predict_onnx import CannotModel
    logger.info("Using ONNX model for predictions.")


class ADBConnectorThread(QThread):
    """
    Worker thread to run loadData.AdbConnector.connect() without blocking the UI.
    """

    connect_finished = pyqtSignal()

    def __init__(self, app: "ArknightsApp"):
        super().__init__()
        self.app = app

    def run(self):
        self.app.adb_connector.connect()
        self.connect_finished.emit()


class HistoryLoader(QThread):
    """
    Worker thread to load HistoryMatch without blocking the UI.
    """

    history_loaded = pyqtSignal(object)

    def run(self):
        history_match = similar_history_match.HistoryMatch()
        # Ensure feat_past and N_history are initialized
        try:
            history_match.feat_past = np.hstack([history_match.past_left, history_match.past_right])
        except Exception:
            history_match.feat_past = None
        history_match.N_history = 0 if history_match.labels is None else len(history_match.labels)

        self.history_loaded.emit(history_match)


class AsyncHistoryMatch(QObject):
    """
    Proxy wrapper for HistoryMatch loaded asynchronously.
    """

    history_loaded = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._match = None
        self._loader = HistoryLoader()
        self._loader.history_loaded.connect(self._on_loaded)
        self._loader.start()

    def _on_loaded(self, history_match):
        self._match = history_match
        # Emit the signal with the fully prepared object
        self.history_loaded.emit(history_match)

    def __getattr__(self, name):
        if self._match is None:
            raise AttributeError(f"HistoryMatch not loaded yet: '{name}'")
        return getattr(self._match, name)


class ArknightsApp(QMainWindow):
    # 添加自定义信号
    update_button_signal = pyqtSignal(str)  # 用于更新按钮文本
    update_monster_signal = pyqtSignal(list)
    update_prediction_signal = pyqtSignal(float)
    update_statistics_signal = pyqtSignal()  # 用于更新统计信息

    def __init__(self):
        super().__init__()
        # 尝试连接模拟器
        self.adb_connector = loadData.AdbConnector()
        self.adb_connector_thread = ADBConnectorThread(self)
        self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
        self.adb_connector_thread.start()

        self.auto_fetch_running = False
        self.no_region = True
        self.first_recognize = True
        self.is_invest = False
        self.game_mode = "单人"

        self.left_monsters = {}
        self.right_monsters = {}
        self.images = {}

        # 模型
        self.cannot_model = CannotModel()

        # 怪物识别模块
        self.recognizer = recognize.RecognizeMonster()

        # 添加历史对局相关属性
        self.history_visible = False
        self.history_data_loaded = False
        self.history_widget = None
        self.history_scroll_area = None

        # 初始化UI后加载历史数据
        self.history_match = AsyncHistoryMatch()
        self.history_match.history_loaded.connect(self.on_history_loaded)

        # 初始化特殊怪物语言触发处理程序
        self.special_monster_handler = SpecialMonsterHandler()

        self.init_ui()
        self.load_monster_data() # 新增：加载怪物数据
        self.load_images()

    def load_monster_data(self):
        self.monster_data = {}
        try:
            with open("monster.csv", "r", encoding='utf-8-sig') as f:
                lines = f.readlines()
                if not lines:
                    logger.warning("monster.csv is empty.")
                    return

                headers = [h.strip() for h in lines[0].strip().split(',')]
                for line in lines[1:]:
                    parts = [p.strip() for p in line.strip().split(',')]
                    if len(parts) >= len(headers):
                        monster_id = parts[0]
                        self.monster_data[monster_id] = {}
                        for i, header in enumerate(headers):
                            self.monster_data[monster_id][header] = parts[i]
        except FileNotFoundError:
            logger.error("monster.csv not found.")
        except Exception as e:
            logger.error(f"Error loading monster data: {e}")

    def init_ui(self):
        try:
            with open("pyproject.toml", "r", encoding="utf-8") as f:
                pyproject_data = toml.load(f)
                version = pyproject_data["project"]["version"]
        except (FileNotFoundError, KeyError):
            version = "unknown"
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.setWindowTitle(f"铁鲨鱼_Arknights Neural Network - v{version} - model: {model_name}")
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
        left_panel.setStyleSheet(
            """
            QWidget#left_panel_id {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            """
        )
        left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        # 人物显示区域
        monster_group = QWidget()
        monster_layout = QVBoxLayout(monster_group)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
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
        """
        )

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
        input_display.setStyleSheet(
            """
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
            """
        )
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
        result_group.setStyleSheet(
            """
            QGroupBox {
                background-color: rgba(120, 120, 120, 10);
                border-radius: 15px;
                border: 1px solid #747474;
            }
            """
        )
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setFont(QFont("Microsoft YaHei", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_label)

        # 添加模型名称显示
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.model_name_label = QLabel(f"model: {model_name}")
        self.model_name_label.setFont(QFont("Microsoft YaHei", 8))
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.model_name_label.setStyleSheet("color: #888888;") # 小字灰色
        result_layout.addWidget(self.model_name_label)

        result_button = QWidget()
        result_button_layout = QHBoxLayout(result_button)

        # 预测按钮 - 带样式
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setStyleSheet(
            """
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
            """
        )
        self.predict_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_entries)
        self.reset_button.setStyleSheet(
            """
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
            """
        )

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

        self.package_data_button = QPushButton("数据打包")
        self.package_data_button.clicked.connect(self.package_data_and_show)
        row1_layout.addWidget(self.package_data_button)

        # 第二行按钮
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)

        self.recognize_button = QPushButton("识别并预测")
        self.recognize_button.clicked.connect(self.recognize_and_predict)
        self.recognize_button.setStyleSheet(
            """
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
            """
        )

        row2_layout.addWidget(self.recognize_button)

        # 第三行按钮
        row3 = QWidget()
        row3_layout = QHBoxLayout(row3)

        self.reselect_button = QPushButton("选择范围")
        self.reselect_button.clicked.connect(self.reselect_roi)
        self.choose_window_button = QPushButton("选择截屏窗口")
        self.choose_window_button.clicked.connect(self.choose_capture_window)

        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QLineEdit()
        self.serial_entry.setFixedWidth(100)
        self.serial_entry.setPlaceholderText("127.0.0.1:5555") # 设置默认灰色文本

        self.serial_button = QPushButton("更新")
        self.serial_button.clicked.connect(self.update_device_serial)

        row3_layout.addWidget(self.choose_window_button)
        row3_layout.addWidget(self.reselect_button)
        row3_layout.addWidget(self.serial_label)
        row3_layout.addWidget(self.serial_entry)
        row3_layout.addWidget(self.serial_button)

        # 第四行按钮 (获取模拟结果)
        row4 = QWidget()
        row4_layout = QHBoxLayout(row4)

        self.simulate_button = QPushButton("沙盒模拟")
        self.simulate_button.clicked.connect(self.run_simulation)
        self.simulate_button.setStyleSheet(
            """
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
            """
        )
        row4_layout.addWidget(self.simulate_button)

        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))

        # 添加所有行到控制布局
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(row3)
        control_layout.addWidget(row4)
        
        # GitHub链接
        github_label = QLabel(
            '<a href="https://github.com/Ancientea/CannotMax" style="color: #2196F3; text-decoration: none;">https://github.com/Ancientea/CannotMax</a>'
        )
        github_label.setAlignment(Qt.AlignmentFlag.AlignLeft) # 改为左对齐
        github_label.setOpenExternalLinks(True)
        github_label.setFont(QFont("Microsoft YaHei", 9))
        github_label.setStyleSheet("padding: 0px; margin: 0px;") # 进一步减少内边距和外边距
        control_layout.addWidget(github_label)

        control_layout.addWidget(self.stats_label)

        right_layout.addWidget(control_group)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(main_widget)

        # 在右侧面板添加历史对局按钮
        self.history_button = QPushButton("显示历史对局")
        self.history_button.clicked.connect(self.toggle_history_panel)
        self.history_button.setStyleSheet(
            """
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
            """
        )
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

        # 连接AutoFetch信号到槽
        self.update_button_signal.connect(self.auto_fetch_button.setText)
        self.update_monster_signal.connect(self.update_monster)
        self.update_prediction_signal.connect(self.update_prediction)
        self.update_statistics_signal.connect(self.update_statistics)

    def on_adb_connected(self):
        logger.info("模拟器初始化完成")

    def on_history_loaded(self, history_match):
        logger.info("尝试获取错题本")
        # Update attributes from loaded HistoryMatch
        self.past_left = history_match.past_left
        self.past_right = history_match.past_right
        self.labels = history_match.labels
        self.feat_past = history_match.feat_past
        self.N_history = history_match.N_history
        self.history_data_loaded = True
        logger.info("错题本加载成功")

    def choose_capture_window(self):
        """弹出窗口选择器，切换 WinRT 截屏源（窗口标题或整屏）。"""
        import traceback, cv2
        if getattr(self, "_switching_source", False):
            return
        self._switching_source = True
        self.choose_window_button.setEnabled(False)
        try:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            dlg = winrt_capture.WindowPickerDialog(self)
            if dlg.exec():
                sel = dlg.get_selection()
                logger.info(f"选择了截屏源: {sel}")
                if not sel:
                    QMessageBox.information(self, "提示", "未选择任何项")
                    return
                hint = ""
                if "window_name" in sel:
                    self.recognizer = recognize.RecognizeMonster(window_name=sel["window_name"], monitor_index=None)
                    hint = f"已切换至窗口：{sel['window_name']}"
                else:
                    idx = max(1, sel["monitor_index"])
                    self.recognizer = recognize.RecognizeMonster(window_name=None, monitor_index=idx)
                    hint = f"已切换至整屏：显示器 {sel['monitor_index']}"
                
                self.no_region = True
                QMessageBox.information(self, "成功", hint + "\n建议重新选择范围。")
        except Exception as e:
            QMessageBox.critical(self, "异常", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._switching_source = False
            self.choose_window_button.setEnabled(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 缩放图片以适应窗口（保持宽高比）
        scaled_pixmap = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # 居中绘制
        painter.drawPixmap(
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap,
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
            shadow01.setColor(QColor(0, 0, 0, 120))  # 发光颜色
            shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
            monster_container.setGraphicsEffect(shadow01)

            monster_container.setStyleSheet("QWidget {border-radius: 0px;}")
            container_layout = QVBoxLayout(monster_container)
            container_layout.setSpacing(2)
            container_layout.setContentsMargins(2, 2, 2, 2)

            # 人物图片
            img_label = QLabel()
            img_label.setFixedSize(60, 60)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            try:
                pixmap = QPixmap(f"images/{i}.png")
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(pixmap)
            except Exception as e:
                logger.error(f"加载人物{i}图片错误: {str(e)}")

            # 添加鼠标悬浮提示
            if str(i) in self.monster_data:
                data = self.monster_data[str(i)]
                tooltip_text = ""
                for key, value in data.items():
                    if key != "id":  # 排除id
                        tooltip_text += f"{key}: {value}\n"
                img_label.setToolTip(tooltip_text.strip())

            # 左输入框
            left_entry = QLineEdit()
            left_entry.setFixedWidth(60)
            left_entry.setPlaceholderText("左")
            left_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.left_monsters[str(i)] = left_entry

            # 右输入框 (放在左输入框下方)
            right_entry = QLineEdit()
            right_entry.setFixedWidth(60)
            right_entry.setPlaceholderText("右")
            right_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.right_monsters[str(i)] = right_entry

            # 添加到容器
            container_layout.addWidget(img_label, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(left_entry, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(right_entry, 0, Qt.AlignmentFlag.AlignCenter)

            # 添加到网格布局
            self.scroll_grid.addWidget(monster_container, row, col, Qt.AlignmentFlag.AlignCenter)

            # 更新行列位置
            col += 1
            if col >= self.COLUMNS:
                col = 0
                row += 1

    def update_input_display(self):
        # 清除现有显示
        for i in reversed(range(self.left_input_layout.count())):
            widget = self.left_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        for i in reversed(range(self.right_input_layout.count())):
            widget = self.right_input_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        left_has_input, right_has_input = False, False
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

        widget.setStyleSheet(
            """
                QWidget {
                    border-radius: 0px;
                }
            """
        )

        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 人物图片
        img_label = QLabel()
        img_label.setFixedSize(70, 70)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        try:
            pixmap = QPixmap(f"images/{monster_id}.png")
            if not pixmap.isNull():
                pixmap = pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                img_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"加载人物{monster_id}图片错误: {str(e)}")
            pass

        # 添加鼠标悬浮提示
        if str(monster_id) in self.monster_data:
            data = self.monster_data[str(monster_id)]
            tooltip_text = ""
            for key, value in data.items():
                tooltip_text += f"{key}: {value}\n"
            img_label.setToolTip(tooltip_text.strip())

        # 数量标签
        count_label = QLabel(count)
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_label.setStyleSheet(
            """
            color: #EDEDED;
            font: bold 20px SimHei;
            border-radius: 5px;
            padding: 2px 5px;
            min-width: 20px;
        """
        )

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
        """更新预测结果显示"""
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
            result_text = (
                f"预测胜方: {winner}\n" f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n"
            )

            # 添加特殊干员提示
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages

        else:
            result_text = (
                f"这一把{winner}\n"
                f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n"
                f"难道说？难道说？难道说？\n"
            )
            self.result_label.setStyleSheet("color: black; font: bold,24px;")

            # 添加特殊干员提示
            special_messages = self.special_monster_handler.check_special_monsters(self, winner)
            if special_messages:
                result_text += "\n" + special_messages

        self.result_label.setText(result_text)

    def predict(self):
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        self.update_input_display()

        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()

    def recognize(self):
        if self.auto_fetch_running:
            screenshot = self.adb_connector.capture_screenshot()
        else:
            screenshot = None

        if self.no_region:  # TODO: 判断需要移至recognize
            if self.first_recognize:
                self.adb_connector.connect()
                self.first_recognize = False
            screenshot = self.adb_connector.capture_screenshot()

        results = self.recognizer.process_regions(screenshot)
        return results, screenshot

    def update_monster(self, results):
        """
        根据识别结果更新怪物面板
        """
        self.reset_entries()
        for res in results:
            if "error" not in res:
                region_id = res["region_id"]
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:
                        entry = self.left_monsters[str(matched_id)]
                    else:
                        entry = self.right_monsters[str(matched_id)]
                    entry.setText(str(number))
                    if entry.text():
                        entry.setStyleSheet("background-color: yellow;")
        self.update_input_display()

    def recognize_and_predict(self):
        results, screenshot = self.recognize()
        self.update_monster(results)
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        # 历史对局
        if self.history_visible and self.history_data_loaded:
            self.render_similar_matches()
        return prediction, results, screenshot

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
        self.history_scroll_area.setStyleSheet(
            """
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
        """
        )

        # 创建内容部件
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

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
            cur_left = np.zeros(MONSTER_COUNT, dtype=float)
            cur_right = np.zeros(MONSTER_COUNT, dtype=float)
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

            title_label.setStyleSheet(
                """
                    QWidget {
                        border-radius: 0px;
                        font-size: 24px;
                        font-weight: bold;
                        color: white;
                    }
                """
            )
            self.history_layout.addWidget(title_label)

            # 渲染每个历史对局
            for idx in top_indices:
                self.add_history_match(idx, sims[idx])

        except Exception as e:
            logger.error(f"渲染历史对局失败: {str(e)}")

    def add_history_match(self, idx, similarity):
        """添加单个历史对局到面板"""
        # 获取历史数据
        left = self.past_left[idx]
        right = self.past_right[idx]
        result = self.labels[idx]

        # 获取当前对局的左右单位
        cur_left = np.zeros(MONSTER_COUNT, dtype=float)
        cur_right = np.zeros(MONSTER_COUNT, dtype=float)
        for name, entry in self.left_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_left[int(name) - 1] = float(v)
        for name, entry in self.right_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_right[int(name) - 1] = float(v)

        # 计算当前对局和历史对局的相似度(不镜像和镜像两种情况)
        setL_cur = set(np.where(cur_left > 0)[0])
        setR_cur = set(np.where(cur_right > 0)[0])
        setL_past = set(np.where(left > 0)[0])
        setR_past = set(np.where(right > 0)[0])

        # 判断是否需要镜像历史对局
        should_swap = len(setL_cur ^ setR_past) + len(setR_cur ^ setL_past) < len(
            setL_cur ^ setL_past
        ) + len(setR_cur ^ setR_past)

        # 创建对局容器
        match_widget = QWidget()
        match_widget.setStyleSheet(
            """
            QWidget {
                background-color: rgba(50, 50, 50, 150);
                border-radius: 10px;
                padding: 0px;
                margin: 5px;
            }
            """
        )
        match_widget.setFixedSize(500, 150)
        match_layout = QVBoxLayout(match_widget)

        # 添加左右阵容
        teams_widget = QWidget()
        teams_layout = QHBoxLayout(teams_widget)

        # 根据是否需要镜像决定显示方向
        if should_swap:
            left_team = self.create_team_widget("右方", right, result == "R")
            right_team = self.create_team_widget("左方", left, result == "L")
        else:
            left_team = self.create_team_widget("左方", left, result == "L")
            right_team = self.create_team_widget("右方", right, result == "R")

        teams_layout.addWidget(left_team)
        teams_layout.addWidget(right_team)
        match_layout.addWidget(teams_widget)

        self.history_layout.addWidget(match_widget)

    def create_team_widget(self, side, counts, is_winner):
        """创建单个队伍显示部件"""
        team_widget = QWidget()
        team_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {'rgba(250, 250, 50, 150)' if is_winner else 'rgba(50, 50, 50, 100)'};
                border-radius: 8px;
                padding: 0px;
                margin: 0px;
            }}
            """
        )

        layout = QVBoxLayout(team_widget)

        # 显示区域
        ops_widget = QWidget()
        shadow01 = QGraphicsDropShadowEffect()
        shadow01.setBlurRadius(5)  # 模糊半径（控制发光范围）
        shadow01.setColor(QColor(0, 0, 0, 120))  # 发光颜色
        shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
        ops_widget.setGraphicsEffect(shadow01)

        ops_widget.setStyleSheet(
            """
                QWidget {
                    background-color: rgba(0, 0, 0, 0);
                    border-radius: 0px;
                    padding: 0px;
                    margin: 0px;
                }
            """
        )
        ops_layout = QHBoxLayout(ops_widget)
        ops_layout.setSpacing(5)
        ops_layout.setContentsMargins(0, 0, 0, 0)

        for i, count in enumerate(counts):
            if count > 0:
                # 创建干员显示
                op_widget = QWidget()
                op_widget.setStyleSheet(
                    "background-color: rgba(0, 0, 0, 0); padding: 0px 0;margin: 0px;"
                )
                op_layout = QVBoxLayout(op_widget)
                op_layout.setContentsMargins(0, 0, 0, 0)
                op_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

                # 干员图片
                img_label = QLabel()
                img_label.setFixedSize(60, 60)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                try:
                    pixmap = QPixmap(f"images/{i + 1}.png")
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        img_label.setPixmap(pixmap)
                except:
                    pass

                # 数量标签
                count_label = QLabel(str(int(count)))
                count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                count_label.setStyleSheet(
                    """
                            color: #EDEDED;
                            font: bold 20px SimHei;
                            min-width: 20px;
                        """
                )

                op_layout.addWidget(img_label, stretch=3)
                op_layout.addWidget(count_label, stretch=1)
                ops_layout.addWidget(op_widget)

        layout.addWidget(ops_widget)
        return team_widget

    def reselect_roi(self):
        self.recognizer.select_roi()
        self.no_region = False

    def toggle_auto_fetch(self):
        if not (hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running):
            self.auto_fetch = auto_fetch.AutoFetch(
                self.adb_connector,
                self.game_mode,
                self.is_invest,
                update_prediction_callback=self.update_prediction_callback,
                update_monster_callback=self.update_monster_callback,
                updater=self.update_statistics_callback,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
                training_duration=float(self.duration_entry.text()) * 3600,  # 获取训练时长
            )
            self.auto_fetch.start_auto_fetch()
        else:
            self.auto_fetch.stop_auto_fetch()

    def update_statistics(self):
        elapsed_time = time.time() - self.auto_fetch.start_time if self.auto_fetch.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.auto_fetch.total_fill_count},    "
            f"填写×次数: {self.auto_fetch.incorrect_fill_count},    "
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟"
        )
        self.stats_label.setText(stats_text)

    def update_device_serial(self):
        new_serial = self.serial_entry.text()
        self.adb_connector.update_device_serial(new_serial)
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {new_serial}")

    def start_callback(self):
        self.update_button_signal.emit("停止自动获取数据")

    def stop_callback(self):
        self.update_button_signal.emit("自动获取数据")

    def update_monster_callback(self, results: list):
        self.update_monster_signal.emit(results)

    def update_prediction_callback(self, prediction: float):
        self.update_prediction_signal.emit(prediction)

    def update_statistics_callback(self):
        self.update_statistics_signal.emit()

    def run_simulation(self):
        """
        获取左右怪物信息，转换为JSON格式，并通过stdin传递给main_sim.py子进程。
        """
        left_monsters_data = {}
        right_monsters_data = {}

        # 获取左侧怪物信息
        for monster_id, entry in self.left_monsters.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                # Need to map monster_id (string) to monster name
                # Assuming MONSTER_MAPPING is accessible or can be imported
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        left_monsters_data[monster_name] = int(count)
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        # 获取右侧怪物信息
        for monster_id, entry in self.right_monsters.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        right_monsters_data[monster_name] = int(count)
                    else:
                        logger.error(f"Monster name not found for ID {monster_id}")
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        simulation_data = {"left": left_monsters_data, "right": right_monsters_data}

        json_data = json.dumps(simulation_data, ensure_ascii=False)
        logger.info(f"Simulation data JSON: {json_data}")

        try:
            # 启动main_sim.py子进程 (非阻塞)
            # Use sys.executable to ensure the same Python interpreter is used
            process = subprocess.Popen(
                [sys.executable, "main_sim.py"], stdin=subprocess.PIPE, text=True, encoding="utf-8"
            )
            # 通过stdin传递JSON数据并关闭stdin
            process.stdin.write(json_data)
            process.stdin.close()
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到 main_sim.py 文件，请检查路径。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动模拟器时发生错误: {str(e)}")

    def get_monster_name_by_id(self, monster_id: int):
        """根据怪物ID获取怪物名称"""
        # Need to import MONSTER_MAPPING from simulator.utils
        try:
            from simulator.utils import MONSTER_MAPPING

            # Adjust for 1-based UI IDs vs 0-based mapping keys
            monster_name = MONSTER_MAPPING.get(monster_id - 1)
            if not monster_name:
                logger.error(f"Monster ID {monster_id} not found in MONSTER_MAPPING.")
            return monster_name
        except ImportError:
            logger.error("Error importing MONSTER_MAPPING from simulator.utils")
            return None

    def update_game_mode(self, mode):
        self.game_mode = mode

    def update_invest_status(self, state):
        self.is_invest = state == Qt.CheckState.Checked

    def update_result(self, text):
        self.result_label.setText(text)

    def update_stats(self, total, incorrect, duration):
        stats_text = f"总共: {total}, 错误: {incorrect}, 时长: {duration}"
        self.stats_label.setText(stats_text)

    def update_image_display(self, qimage):
        self.image_display.setPixmap(QPixmap.fromImage(qimage).scaled(
            self.image_display.width(),
            self.image_display.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))


    def package_data_and_show(self):
        try:
            zip_filename = data_package.package_data()
            if zip_filename:
                # 在文件浏览器中高亮显示文件
                subprocess.run(f'explorer /select,"{zip_filename}"')
                QMessageBox.information(self, "成功", f"数据已打包到 {zip_filename}")
            else:
                QMessageBox.warning(self, "警告", "没有找到可以打包的数据目录。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打包数据时发生错误: {str(e)}")


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec()
