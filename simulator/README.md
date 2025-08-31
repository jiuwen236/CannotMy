# 明日方舟斗蛐蛐模拟器 🎮⚔️

这是一款基于明日方舟角色设定的AI自动战斗模拟器，还原干员技能特性与战场策略对抗，并且可以以极快速度获得战斗结果！

---

## 🌟 核心特色

- **全自动战斗系统**：基于属性与AI策略的自主决策
- **精细角色建模**：
  - 完全还原明日方舟斗蛐蛐的敌人属性设定
  - 真实攻击范围计算（可配置十字/圆形/直线范围）
- **动态战斗表现**：
  - 实时战斗日志
  - 可视化战场轨迹
  - 数据统计面板

---

## ⚙️ 安装指南

### 环境要求
- Python 3.8+
- NumPy 1.20+


## 🕹️ 使用说明

### 启动战斗

**示例**：

具体可以参考 `simulate.py`
```python
from battle_field import Battlefield, Faction

scene_config = { "left": { "Vvan": 5 }, "right": { "衣架": 13 }, "result": "right" }

# 用户配置
left_army = scene_config["left"]
right_army = scene_config["right"]
    
# 初始化战场
battlefield = Battlefield(monster_data)

# 去除掉不符合格式的配置
if not battlefield.setup_battle(left_army, right_army, monster_data):
    # 处理异常

# 开始战斗，并且返回胜者，可以打开visualize模式来可视化结果
winner = battlefield.run_battle(visualize=VISUALIZATION_MODE)

```

### 参数说明
---

## 📝 配置文件规范

### 敌人模板
```json
    {
      "名字": "绵羊",
      "攻击力": { "数值": 570 },
      "物理防御": { "数值": 250 },
      "生命值": { "数值": 3500 },
      "类型": "魔法",
      "法抗": { "数值": 50 },
      "攻击间隔": { "数值": 3.2 },
      "移速": { "数值": 0.7 },
      "攻击范围": { "数值": 3.5 },
      "特性": ""
    },
    {
      "名字": "食腐狗",
      "攻击力": { "数值": 825 },
      "物理防御": { "数值": 150 },
      "生命值": { "数值": 6000 },
      "类型": "物理",
      "法抗": { "数值": 20 },
      "攻击间隔": { "数值": 1.8 },
      "移速": { "数值": 1.4 },
      "攻击范围": { "数值": 0.8 },
      "特性": "每次攻击令其在10秒内每秒受到100点魔法伤害"
    },
```

### 队伍配置
可以参考`scene.json`文件的内容
```json
{
    "left":
    [
        { "name": "酸液源石虫·α", "count": 0 },
        { "name": "高能源石虫", "count": 0 },
        { "name": "阿咬", "count": 0 },
        { "name": "狂暴的猎狗pro", "count": 0 },
        { "name": "提亚卡乌好战者", "count": 0 },
        { "name": "污染躯壳", "count": 0 },
        { "name": "大盾哥", "count": 0 },
        { "name": "宿主流浪者", "count": 0 },
        { "name": "光剑", "count": 4 },
        { "name": "泥岩巨像", "count": 0 },
        { "name": "呼啸骑士团学徒", "count": 0 },
        { "name": "萨卡兹大剑手", "count": 0 },
        { "name": "狂暴宿主组长", "count": 0 },
        { "name": "海螺", "count": 0 },
        { "name": "拳手囚犯", "count": 0 },
        { "name": "高塔术师", "count": 0 },
        { "name": "冰原术师", "count": 0 },
        { "name": "矿脉守卫", "count": 0 },
        { "name": "“庞贝”", "count": 0 },
        { "name": "绵羊", "count": 0 },
        { "name": "食腐狗", "count": 0 },
        { "name": "温顺的武装驮兽", "count": 3 },
        { "name": "鼠鼠", "count": 4 },
        { "name": "“投石机”", "count": 0 },
        { "name": "船长", "count": 0 }
    ],
    "right":
    [
        { "name": "酸液源石虫·α", "count": 0 },
        { "name": "高能源石虫", "count": 0 },
        { "name": "阿咬", "count": 0 },
        { "name": "狂暴的猎狗pro", "count": 0 },
        { "name": "提亚卡乌好战者", "count": 0 },
        { "name": "污染躯壳", "count": 0 },
        { "name": "大盾哥", "count": 0 },
        { "name": "宿主流浪者", "count": 0 },
        { "name": "光剑", "count": 0 },
        { "name": "泥岩巨像", "count": 0 },
        { "name": "呼啸骑士团学徒", "count": 0 },
        { "name": "萨卡兹大剑手", "count": 0 },
        { "name": "狂暴宿主组长", "count": 0 },
        { "name": "海螺", "count": 0 },
        { "name": "拳手囚犯", "count": 11 },
        { "name": "高塔术师", "count": 0 },
        { "name": "冰原术师", "count": 0 },
        { "name": "矿脉守卫", "count": 0 },
        { "name": "“庞贝”", "count": 3 },
        { "name": "绵羊", "count": 9 },
        { "name": "食腐狗", "count": 0 },
        { "name": "温顺的武装驮兽", "count": 0 },
        { "name": "鼠鼠", "count": 0 },
        { "name": "“投石机”", "count": 0 },
        { "name": "船长", "count": 0 }
    ]
}
```

---

## 🧠 核心机制

### 目标选择算法

### 战斗流程


## 🤝 参与贡献

欢迎提交PR！

---

## 📜 许可证

本项目采用 [MIT License](LICENSE)，可自由用于学习与二次创作。