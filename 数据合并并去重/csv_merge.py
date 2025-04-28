import os
import csv


def read_csv_data(filename):
    """读取CSV文件，自动检测编码并验证表头是否为纯数字，返回表头、数据行（去重后的集合）和成功的编码"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']  # 尝试的编码顺序
    for encoding in encodings:
        try:
            with open(filename, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(f"文件 {filename} 为空或无法读取内容")
                if all(field.strip().isdigit() for field in header):
                    data = {tuple(row) for row in reader}
                    return header, data, encoding
                else:
                    continue  # 表头不符合要求，尝试下一编码
        except UnicodeDecodeError:
            continue  # 解码失败，尝试下一编码
    raise ValueError(f"无法以支持表头为纯数字的编码读取文件 {filename}")


# 获取当前文件夹下的所有CSV文件
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

if not csv_files:
    raise ValueError("当前文件夹下没有找到CSV文件")

# 读取所有CSV文件并验证表头一致性
all_data = set()
header = None

for file in csv_files:
    try:
        file_header, file_data, file_encoding = read_csv_data(file)
        if header is None:
            header = file_header
        elif header != file_header:
            raise ValueError(f"错误：文件 {file} 的表头与其他文件不一致，无法合并")
        all_data |= file_data  # 合并数据
        print(f"文件 {file} 使用的编码是: {file_encoding}")
    except ValueError as e:
        print(f"错误: {e}")
        exit()

# 自动生成表头（从1开始的数字序列）
generated_header = list(range(1, len(header) + 1))

# 写入合并后的文件
output_file = 'arknights.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(generated_header)  # 写入表头
    for row in all_data:
        writer.writerow(row)

print(f"所有CSV文件合并完成，结果已保存到 {output_file}")