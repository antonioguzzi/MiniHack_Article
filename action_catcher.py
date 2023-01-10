import numpy as np


def guess_movements(pre_p, curr_p):  # 0 = y, 1 = x
    if curr_p[0] == pre_p[0] - 1 and curr_p[1] == pre_p[1] - 1:
        return 7  # MOVE_ACTIONS[7]  # "↖"
    if curr_p[0] == pre_p[0] + 1 and curr_p[1] == pre_p[1] + 1:
        return 5  # MOVE_ACTIONS[5]  # "↘"
    if curr_p[0] == pre_p[0] - 1 and curr_p[1] == pre_p[1] + 1:
        return 4  # MOVE_ACTIONS[4]  # "↗"
    if curr_p[0] == pre_p[0] + 1 and curr_p[1] == pre_p[1] - 1:
        return 6  # MOVE_ACTIONS[6]  # "↙"
    if curr_p[1] == pre_p[1] + 1:
        return 1  # MOVE_ACTIONS[1]  # "→"
    if curr_p[1] == pre_p[1] - 1:
        return 3  # MOVE_ACTIONS[3]  # "←"
    if curr_p[0] == pre_p[0] - 1:
        return 0  # MOVE_ACTIONS[0]  # "↑"
    if curr_p[0] == pre_p[0] + 1:
        return 2  # MOVE_ACTIONS[2]  # "↓"
    return None
