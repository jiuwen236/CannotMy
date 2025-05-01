import os
import shutil
import zipfile

def create_zip_package(output_zip_path):
    # 定义文件和文件夹路径
    csv_file = "arknights.csv"
    data_folder = "data"
    images_folder = os.path.join(data_folder, "images")

    # 检查文件和文件夹是否存在
    if not os.path.exists(csv_file):
        print(f"文件 {csv_file} 不存在！")
        return
    if not os.path.exists(images_folder):
        print(f"文件夹 {images_folder} 不存在！")
        return

    # 创建压缩包
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 添加 arknights.csv
        zipf.write(csv_file, arcname="arknights.csv")

        # 添加 data/images 文件夹及其内容
        for root, _, files in os.walk(images_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # 在压缩包中保留 data/images 的目录结构
                arcname = os.path.relpath(file_path, start=data_folder)
                zipf.write(file_path, arcname=arcname)

    print(f"压缩包已创建：{output_zip_path}")

# 调用函数创建压缩包
create_zip_package("arknights_package.zip")