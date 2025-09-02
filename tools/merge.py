import os
import glob
import csv
from collections import defaultdict, Counter

# 允许收集任意重复行(高自重除外)
ALLOW_DUPLICATES = False

# 是否使用整行参与重复检测（包含非数据部分）(不做其他额外检测了)
USE_FULL_ROW_FOR_DUP = False
# 输出是否包含非数据部分
INCLUDE_NON_DATA_IN_OUTPUT = False
# 当本开关为 True 且 USE_FULL_ROW_FOR_DUP = True 时，要求所有文件的数据行列数完全相同（比如都是156列），否则抛出错误
ENFORCE_SAME_COLS_WHEN_FULL_ROW = False

# 获取指定目录下所有的csv文件（递归查找）
output_file='arknights.csv'
csv_files = glob.glob('./**/*.csv', recursive=True)

LOG = False
DUPLICATE_THRESHOLD = 0.1

# 仅用于数据部分截断的上限
DATA_COL_LIMIT = 155

duplicate_cnt = 0

# 将一行规范化为“去重键”（是否包含非数据部分由开关控制）
def _key_from_row(row):
    return tuple(row) if USE_FULL_ROW_FOR_DUP else tuple(row[:DATA_COL_LIMIT])

# 根据配置决定输出整行还是仅数据部分
def _row_for_output(row):
    return tuple(row) if INCLUDE_NON_DATA_IN_OUTPUT else _key_from_row(row)

# 新增：获取“跳过首行后”的首个非空数据行列数
def _first_data_len_after_first_line(filename):
    with open(filename, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader, None)  # 跳过文件首行（无论是否为表头）
        for row in reader:
            if row and len(row) > 0:
                return len(row)
    return None

def _read_file_rows(filename):
    """读取文件，按顺序返回整行(tuple(row))。首行若为表头则跳过"""
    # print(f"读取文件: {filename}")
    rows = []
    with open(filename, 'r', encoding='utf-8') as infile:
        csv_reader = csv.reader(infile)
        first_row = next(csv_reader, None)
        if first_row and len(first_row) > 0 and ((first_row[0] == "1L") or (first_row[0] == "1" and len(first_row) > 1 and first_row[1] == "2")):
            # print(f"{filename} 有表头，跳过第一行")
            pass
        else:
            if first_row and len(first_row) > 0:
                rows.append(tuple(first_row))  # 保留整行

        # 启用（从文件第二行开始的）列数一致性检查
        enforce_cols = ENFORCE_SAME_COLS_WHEN_FULL_ROW and USE_FULL_ROW_FOR_DUP
        expected_len = None  # 来自第二行（或首个非空数据行）的列数
        line_no = 1  # 已经读取过第一行

        for row in csv_reader:
            line_no += 1
            if row and len(row) > 0:
                if enforce_cols:
                    if expected_len is None:
                        expected_len = len(row)  # 以第二行（或首个非空行）的列数为基准
                    elif len(row) != expected_len:
                        raise ValueError(f"文件 {filename} 第{line_no} 行列数不一致：期望 {expected_len} 列，实际 {len(row)} 列。")
                rows.append(tuple(row))  # 保留整行
    return rows

def process_csv_files():
    if not csv_files:
        print("目录下没有找到CSV文件")
        return
    print(f"正在处理以下CSV文件: {csv_files}")

    # 快速分支：当要求用整行参与去重时，直接全局去重（严格不允许任何重复），
    # 不做同源关系或文件内自重的复杂检测。
    if USE_FULL_ROW_FOR_DUP:
        seen = set()
        collected_rows = []
        for filename in csv_files:
            rows = _read_file_rows(filename)
            for row in rows:
                key = tuple(row)  # 整行作为去重键
                if key in seen:
                    continue
                seen.add(key)
                collected_rows.append(_row_for_output(row))
        # 写入输出并结束
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            for row in collected_rows:
                csv_writer.writerow(row)
        print(f"处理完成（整行去重模式），共有{len(collected_rows)}行唯一数据，结果保存在 {output_file}")
        return

    # 原有在 USE_FULL_ROW_FOR_DUP == False 时的复杂逻辑保持不变

    # 新增：全局列数一致性检查（从每个文件第二行起）
    if ENFORCE_SAME_COLS_WHEN_FULL_ROW and USE_FULL_ROW_FOR_DUP:
        file_len_map = {}
        for fname in csv_files:
            l = _first_data_len_after_first_line(fname)
            if l is not None:
                file_len_map[fname] = l
        unique_lens = set(file_len_map.values())
        if len(unique_lens) > 1:
            details = ", ".join(f"{os.path.basename(k)}:{v}" for k, v in sorted(file_len_map.items()))
            raise ValueError(f"所有文件的数据列数（自第二行起）不一致：{details}。请确保列数相同或关闭 ENFORCE_SAME_COLS_WHEN_FULL_ROW。")

    # 预读取所有文件，保留每个文件的整行
    file_rows = {}
    for filename in csv_files:
        file_rows[filename] = _read_file_rows(filename)

    # 构建每个文件的去重键集合与计数（根据配置决定是否包含非数据部分）
    file_to_keys_all = {fname: set(_key_from_row(r) for r in rows) for fname, rows in file_rows.items()}

    # 构建每个文件内“键 -> 次数”统计（用于加权选择）
    file_to_key_count = {fname: Counter(_key_from_row(r) for r in rows) for fname, rows in file_rows.items()}

    # 统计每个文件的自重复率，并决定是否将“n条i按1条计”
    file_dup_rate = {}
    collapse_intra_file = {}
    for fname in csv_files:
        total = len(file_rows.get(fname, []))
        unique_cnt = len(file_to_keys_all.get(fname, set()))
        dup_rate = 0.0 if total == 0 else (1.0 - unique_cnt / total)
        file_dup_rate[fname] = dup_rate
        collapse_intra_file[fname] = (dup_rate > DUPLICATE_THRESHOLD)
        # 可视化输出
        if LOG:
            print(f"{fname} 自重复率: {dup_rate:.2%} -> {'按1计' if collapse_intra_file[fname] else '按原始计数'}")

    # 计算文件级“同源关系”邻接表（基于 DUPLICATE_THRESHOLD 阈值）
    same_origin_adj = {fname: set() for fname in csv_files}
    n_files = len(csv_files)
    for i in range(n_files):
        fA = csv_files[i]
        keysA = file_to_keys_all.get(fA, set())
        lenA = len(keysA)
        if lenA == 0:
            continue
        for j in range(i + 1, n_files):
            fB = csv_files[j]
            keysB = file_to_keys_all.get(fB, set())
            lenB = len(keysB)
            if lenB == 0:
                continue
            inter = keysA & keysB
            if not inter:
                continue
            pctA = len(inter) / lenA
            pctB = len(inter) / lenB
            if pctA >= DUPLICATE_THRESHOLD or pctB >= DUPLICATE_THRESHOLD:
                same_origin_adj[fA].add(fB)
                same_origin_adj[fB].add(fA)

    # 构建 “键 -> 出现该键的文件集合”
    key_to_files = defaultdict(set)
    for fname, keys in file_to_keys_all.items():
        for k in keys:
            key_to_files[k].add(fname)

    # 为每个键计算 isolated_files_per_key 与 selected_files_per_key
    isolated_files_per_key = {}
    selected_files_per_key = {}

    for key, files in key_to_files.items():
        files_list = list(files)
        neighbors_in_key = {f: (same_origin_adj.get(f, set()) & files) for f in files_list}
        iso_files = {f for f in files_list if not neighbors_in_key[f]}
        isolated_files_per_key[key] = iso_files

        conflict_nodes = [f for f in files_list if neighbors_in_key[f]]
        if conflict_nodes:
            # 权重：每个文件内该键的出现次数；若文件自重复率超阈值，则此键在该文件内按1计
            weights = {}
            for f in conflict_nodes:
                base = file_to_key_count[f][key]
                weights[f] = 1 if collapse_intra_file.get(f, False) and base > 0 else base
            # 高权重优先，度数次之，最后按文件名稳定排序
            sorted_nodes = sorted(conflict_nodes, key=lambda f: (-weights[f], len(neighbors_in_key[f]), f))
            selected = set()
            banned = set()
            for f in sorted_nodes:
                if f not in banned:
                    selected.add(f)
                    banned.update(neighbors_in_key[f])
            selected_files_per_key[key] = selected
        else:
            selected_files_per_key[key] = set()

    # 收集结果；忽略策略：
    # - 同源冲突子图：仅保留被选中文件的全部行；未选中的全部忽略
    # - 非同源（度=0）文件：保留全部出现
    collected_rows = []
    same_origin_ignored_cnt = 0
    non_ignored_dup_cnt = 0

    # 用于统计“未被忽略的重复数据数量”：仅与之前处理过的文件比较
    seen_in_prev_files = set()

    # 按发现顺序处理文件
    for filename in csv_files:
        rows = file_rows[filename]  # 整行
        # 当不允许任意重复行时，针对自重复率超阈值的文件：同一键在该文件内按1条计
        collapse = collapse_intra_file.get(filename, False)
        seen_keys_in_file = set() if (not ALLOW_DUPLICATES and collapse) else None

        if ALLOW_DUPLICATES:
            for row in rows:
                collected_rows.append(_row_for_output(row))
        else:
            for row in rows:
                key = _key_from_row(row)
                iso_files = isolated_files_per_key.get(key, set())
                if filename in iso_files:
                    # 非同源：保留；若该文件需要按1条计，则仅保留该键在本文件内的首条
                    if seen_keys_in_file is not None:
                        if key in seen_keys_in_file:
                            continue
                        seen_keys_in_file.add(key)
                    collected_rows.append(_row_for_output(row))
                    if key in seen_in_prev_files:
                        non_ignored_dup_cnt += 1
                else:
                    # 同源冲突：仅保留被选中文件的行；若该文件需要按1条计，则仅保留首条
                    selected_files = selected_files_per_key.get(key, set())
                    if filename in selected_files:
                        if seen_keys_in_file is not None:
                            if key in seen_keys_in_file:
                                continue
                            seen_keys_in_file.add(key)
                        collected_rows.append(_row_for_output(row))
                        if key in seen_in_prev_files:
                            non_ignored_dup_cnt += 1
                    else:
                        same_origin_ignored_cnt += 1

        # 当前文件的键加入“已出现键”集合（用于非同源重复计数）
        seen_in_prev_files.update(file_to_keys_all.get(filename, set()))

    # 创建输出文件并写入
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        for row in collected_rows:
            csv_writer.writerow(row)

    if ALLOW_DUPLICATES:
        print(f"处理完成，共有{len(collected_rows)}行数据，结果保存在 {output_file}")
    else:
        print(f"处理完成，共有{len(collected_rows)}行数据（同源冲突按权重保留；非同源重复保留多条；高自重复文件同键按1计），结果保存在 {output_file}。")
        print(f"有 {same_origin_ignored_cnt} 行同源重复数据被忽略（未被选中的同源文件中的行）。")
        print(f"有 {non_ignored_dup_cnt} 行重复数据未被忽略（非同源重复）。")

if __name__ == '__main__':
    process_csv_files()