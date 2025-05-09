def calculate_damage(attacker, defender):
    config = attacker.config
    base_attack = config["attack"]

    # 火刀效果处理（放在伤害计算前）
    if config.get("effect") == "火刀":
        # 获取最大生命值的50%
        half_health = attacker.config["health"] * 0.5
        # 当前生命值低于50%时触发
        if attacker.health <= half_health:
            base_attack = base_attack * 2.8  # 提升至280%

    # 物理伤害
    if config["damage_type"] == "物理":
        raw_damage = base_attack - defender.current_defense
        # 保底伤害使用当前攻击力计算
        final_damage = max(base_attack * 0.05, raw_damage)

    # 法术伤害
    if config["damage_type"] == "法术":
        resistance = defender.config["magic_resist"]
        multiplier = (100 - resistance) / 100
        raw_damage = config["attack"] * multiplier
        final_damage = max(config["attack"] * 0.05, raw_damage)

    # 破甲效果（测过了，先造成伤害再破甲）
    if config.get("effect") == "破甲15":
        # 防御值最多降到0
        new_defense = max(defender.current_defense - 15, defender.min_defense)
        defender.current_defense = new_defense
    if config.get("effect") == "破甲10":
        # 防御值最多降到0
        new_defense = max(defender.current_defense - 10, defender.min_defense)
        defender.current_defense = new_defense

    # 灼燃效果处理（仅在非冷却状态时叠加）
    if config.get("effect") == "灼燃" and defender.burn_cooldown <= 0:
        defender.burn_damage -= config["attack"]  # 减少灼燃损伤条

        # 触发灼燃爆发
        if defender.burn_damage <= 0:
            # 法抗降低（不低于0）
            defender.current_magic_resist = max(defender.original_magic_resist - 20, 0)
            defender.burn_effect_duration = 10  # 10秒持续时间

            # 7000元素伤害
            final_damage = 7000

            # 重置状态
            defender.burn_cooldown = 10  # 10秒冷却
            defender.burn_damage = 1000

    return final_damage