import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl


def build_fuzzy_system():
    t_pause = ctrl.Antecedent(
        np.arange(0, 11, 1),
        "pause_time"
    )

    f_err = ctrl.Antecedent(
        np.arange(0, 6, 1),
        "error_count"
    )

    c_level = ctrl.Consequent(
        np.arange(0, 1.01, 0.01),
        "confusion_level"
    )

    t_pause["short"] = fuzzy.trimf(
        t_pause.universe,
        [0, 0, 4]
    )

    t_pause["medium"] = fuzzy.trimf(
        t_pause.universe,
        [2, 5, 8]
    )

    t_pause["long"] = fuzzy.trimf(
        t_pause.universe,
        [6, 10, 10]
    )

    f_err["low"] = fuzzy.trimf(
        f_err.universe,
        [0, 0, 2]
    )

    f_err["medium"] = fuzzy.trimf(
        f_err.universe,
        [1, 2.5, 4]
    )

    f_err["high"] = fuzzy.trimf(
        f_err.universe,
        [3, 5, 5]
    )

    c_level["low"] = fuzzy.trimf(
        c_level.universe,
        [0, 0, 0.4]
    )

    c_level["medium"] = fuzzy.trimf(
        c_level.universe,
        [0.3, 0.5, 0.7]
    )

    c_level["high"] = fuzzy.trimf(
        c_level.universe,
        [0.6, 1.0, 1.0]
    )

# Fuzzy Rule Base
# 覆盖 pause × error 的主要组合

    # 停顿短 + 错误少 → 困惑低
    rule1 = ctrl.Rule(
        t_pause["short"] & f_err["low"],
        c_level["low"]
    )

    # 停顿短 + 错误中等 → 困惑中等
    rule2 = ctrl.Rule(
        t_pause["short"] & f_err["medium"],
        c_level["medium"]
    )

    # 停顿短 + 错误很多 → 困惑高
    rule3 = ctrl.Rule(
        t_pause["short"] & f_err["high"],
        c_level["high"]
    )

    # 停顿中等 + 错误少 → 困惑中等
    rule4 = ctrl.Rule(
        t_pause["medium"] & f_err["low"],
        c_level["medium"]
    )

    # 停顿中等 + 错误中等 → 困惑中等
    rule5 = ctrl.Rule(
        t_pause["medium"] & f_err["medium"],
        c_level["medium"]
    )

    # 停顿中等 + 错误很多 → 困惑高
    rule6 = ctrl.Rule(
        t_pause["medium"] & f_err["high"],
        c_level["high"]
    )

    # 停顿很长时，即使错误少，也可能已经卡住
    rule7 = ctrl.Rule(
        t_pause["long"] & f_err["low"],
        c_level["medium"]
    )

    # 停顿长 + 错误中等 → 高困惑
    rule8 = ctrl.Rule(
        t_pause["long"] & f_err["medium"],
        c_level["high"]
    )

    # 停顿长 + 错误多 → 高困惑
    rule9 = ctrl.Rule(
        t_pause["long"] & f_err["high"],
        c_level["high"]
    )

    system = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
    rule9
])

    return ctrl.ControlSystemSimulation(system)


def calculate_confusion(
    pause_time: float,
    error_count: int
):
    # 防止前端传入超出 fuzzy universe 的值
    pause_time = max(0.0, min(float(pause_time), 10.0))
    error_count = max(0, min(int(error_count), 5))

    simulator = build_fuzzy_system()

    simulator.input["pause_time"] = pause_time
    simulator.input["error_count"] = error_count

    simulator.compute()

    # 防止极端情况下没有产生输出
    if "confusion_level" not in simulator.output:
        return 0.5

    return float(
        simulator.output["confusion_level"]
    )