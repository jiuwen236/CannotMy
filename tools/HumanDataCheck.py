import tkinter as tk
from PIL import Image, ImageTk
import csv
import os
from recognize import MONSTER_COUNT


class ArknightsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arknights Data Viewer")
        self.root.geometry("800x300")

        # 绑定快捷键
        self.root.bind("<Left>", lambda event: self.show_prev_row())  # 小键盘左键
        self.root.bind("<Right>", lambda event: self.show_next_row())  # 小键盘右键
        self.root.bind("<Delete>", lambda event: self.delete_current_row())  # 删除键

        # 创建顶部和底部框架
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=10)
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(pady=10)

        # 创建按钮
        self.next_button = tk.Button(root, text="下一个", command=self.show_next_row)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.prev_button = tk.Button(root, text="上一个", command=self.show_prev_row)
        self.prev_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.delete_button = tk.Button(root, text="删除数据", command=self.delete_current_row)
        self.delete_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # 添加行号显示和跳转功能
        self.row_label = tk.Label(root, text="当前行号: 0")
        self.row_label.pack(side=tk.LEFT, padx=10)

        self.row_entry = tk.Entry(root, width=5)
        self.row_entry.pack(side=tk.LEFT, padx=5)

        self.jump_button = tk.Button(root, text="跳转", command=self.jump_to_row)
        self.jump_button.pack(side=tk.LEFT, padx=5)

        # 初始化数据
        self.data = self.read_csv("arknights.csv")
        self.current_row_index = 0

        # 加载图片
        self.images = self.load_all_images()

        # 显示第一行数据
        self.show_row(self.current_row_index)

    def read_csv(self, file_path):
        """读取 CSV 文件"""
        data = []
        with open(file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        return data[1:]  # 跳过表头

    def load_all_images(self):
        """加载所有图片"""
        images = {}
        for i in range(1, MONSTER_COUNT + 1):  # 使用 MONSTER_COUNT 动态加载图片
            image_path = os.path.join("images", f"{i}.png")
            if os.path.exists(image_path):
                image = Image.open(image_path).resize((50, 50))
                images[str(i)] = ImageTk.PhotoImage(image)
            else:
                print(f"Image {i}.png not found.")
                images[str(i)] = None  # 占位符
        return images

    def show_row(self, row_index):
        """显示指定行的非0列数据"""
        for widget in self.top_frame.winfo_children():
            widget.destroy()  # 清空顶部框架内容

        if row_index >= len(self.data):
            return

        row = self.data[row_index]

        # 显示左方怪物
        for i in range(1, MONSTER_COUNT + 1):
            value = row[i - 1]
            try:
                value = float(value)
                if value > 0:  # 仅显示非0值
                    tk.Label(self.top_frame, image=self.images[str(i)]).grid(row=0, column=i - 1)
                    tk.Label(self.top_frame, text=str(int(value))).grid(row=1, column=i - 1)
            except ValueError:
                print(f"Skipping invalid value: {row[i - 1]} at column {i - 1}")

        # 插入空白间隔
        gap_column = MONSTER_COUNT-1  # 间隔列索引
        tk.Label(self.top_frame, text="").grid(row=0, column=gap_column, padx=50)  # 添加水平间距

        # 显示右方怪物
        for i in range(MONSTER_COUNT + 1, MONSTER_COUNT*2 + 1):
            value = row[i - 1]
            try:
                value = float(value)
                if value > 0:  # 仅显示非0值
                    tk.Label(self.top_frame, image=self.images[str(i - MONSTER_COUNT)]).grid(row=0, column=i - 1)
                    tk.Label(self.top_frame, text=str(int(value))).grid(row=1, column=i - 1)
            except ValueError:
                print(f"Skipping invalid value: {row[i - 1]} at column {i - 1}")

        # 显示 L/R
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()  # 清空底部框架内容
        tk.Label(self.bottom_frame, text=f"L/R: {row[MONSTER_COUNT*2]}", font=("Arial", 16)).pack()

        # 显示最后一列的图片
        final_image_path = os.path.join("data", "images", row[-1])
        if os.path.exists(final_image_path):
            final_image = Image.open(final_image_path)
            final_image_tk = ImageTk.PhotoImage(final_image)
            tk.Label(self.bottom_frame, image=final_image_tk).pack()
            self.bottom_frame.image = final_image_tk  # 防止图片被垃圾回收

        # 更新行号显示
        self.row_label.config(text=f"当前行号: {row_index + 1}")

    def jump_to_row(self):
        """跳转到指定行"""
        try:
            row_index = int(self.row_entry.get()) - 1  # 转换为索引
            if 0 <= row_index < len(self.data):
                self.current_row_index = row_index
                self.show_row(self.current_row_index)
            else:
                print("行号超出范围")
        except ValueError:
            print("请输入有效的行号")

    def show_prev_row(self):
        """显示上一行数据"""
        if self.current_row_index > 0:
            self.current_row_index -= 1
        else:
            self.current_row_index = len(self.data) - 1  # 跳转到最后一行
        self.show_row(self.current_row_index)

    def show_next_row(self):
        """显示下一行数据"""
        if self.current_row_index < len(self.data) - 1:
            self.current_row_index += 1
        else:
            self.current_row_index = 0  # 跳转到第一行
        self.show_row(self.current_row_index)

    def delete_current_row(self):
        """删除当前行数据"""
        if self.current_row_index < len(self.data):
            # 获取当前行最后一列的图片路径
            image_name = self.data[self.current_row_index][-1]
            image_path = os.path.join("data", "images", image_name)

            # 删除图片文件（如果存在）
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"已删除图片文件: {image_path}")
            else:
                print(f"图片文件不存在: {image_path}")

            del self.data[self.current_row_index]  # 从内存中删除当前行
            # 将修改后的数据写回 CSV 文件
            with open("arknights.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [f"{i}L" for i in range(1, MONSTER_COUNT + 1)] +
                    [f"{i}R" for i in range(1, MONSTER_COUNT + 1)] +
                    [f"{MONSTER_COUNT * 2 + 1}"]
                )
                writer.writerows(self.data)
            # 更新显示
            if self.current_row_index >= len(self.data):
                self.current_row_index -= 1  # 防止越界
            self.show_row(self.current_row_index)


# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = ArknightsApp(root)
    root.mainloop()