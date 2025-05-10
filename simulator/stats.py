import json
from collections import defaultdict


data = """
{"left": {"冰原术师": 16}, "right": {"雪境精锐": 4, "小锤": 1}, "result": "right"}
"""
with open("errors.json", encoding='utf-8') as f:
    data = f.read()

with open("all.json", encoding='utf-8') as f:
    data2 = f.read()
counter = defaultdict(int)
power_counter = defaultdict(int)
counter2 = defaultdict(int)

for line in data.strip().split('\n'):
    record = json.loads(line)
    result = record['result']
    for side in ["left", "right"]:
        for monster, count in record[side].items():
            counter[monster] += 1
            if side == result:
                power_counter[monster] += 1
            else:
                power_counter[monster] -= 1

index = 0
for line in data2.strip().split('\n'):
    index+=1
    record = json.loads(line)
    for side in ["left", "right"]:
        for monster, count in record[side].items():
            counter2[monster] += 1
    if index == 12168:
        break


sorted_monsters = sorted(counter.items(), key=lambda x: (-x[1] / counter2[x[0]], x[0]))

for monster, count in sorted_monsters:
    str = "低估" if power_counter[monster] > 0 else "高估"
    print(f"{monster}: {count}/{counter2[monster]} = {count/counter2[monster]} 估计值:【{str}】{-power_counter[monster] / count}")