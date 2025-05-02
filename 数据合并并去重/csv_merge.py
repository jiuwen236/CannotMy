import csv
import os


def read_csv_data(filename):
    """读取CSV文件，自动检测编码并返回表头、数据行（去重后的集合）和成功的编码"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']  # 尝试的编码顺序
    for encoding in encodings:
        try:
            with open(filename, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(f"文件 {filename} 为空或无法读取内容")
                data = {tuple(row) for row in reader}
                return header, data, encoding
        except UnicodeDecodeError:
            continue  # 解码失败，尝试下一编码
    raise ValueError(f"无法以支持的编码读取文件 {filename}")


# 获取当前文件夹下的所有CSV文件
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

if not csv_files:
    print("当前文件夹下没有找到CSV文件")
    exit()

merged_data = set()
header = None

# 读取所有CSV文件并合并数据
for csv_file in csv_files:
    try:
        current_header, data, encoding = read_csv_data(csv_file)
        if header is None:
            header = current_header
        elif header != current_header:
            print(f"警告：文件 {csv_file} 的表头与其他文件不一致，可能会导致数据不匹配")
        merged_data |= data
        print(f"文件 {csv_file} 使用的编码是: {encoding}")
    except ValueError as e:
        print(f"错误: {e}")
        exit()

# 写入合并后的文件
with open('arknights.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in merged_data:
        writer.writerow(row)

print(f"所有CSV文件合并完成，结果已保存到 arknights.csv")