# ring_auto.py - 生成 ring 模擬數據
import numpy as np
import time
import meep as mp  # 先導入 meep 確保 Medium 可用
from ring_fuc import ring_run_simulation  # 引入模擬函數
import random

# =============================================================================
# 參數範圍設定（與 ring_regression_study 對應）
# =============================================================================

# 生成樣本數
N_SAMPLES = 100

# resolution: 空間分辨率 (pixels/μm)，影響模擬精度
# 可選 [51, 71, 81, 101] 以增加多樣性，或固定單值加速
RESOLUTION_OPTIONS = [62,55,72,60,45,67,51]#, 81, 61]

# period 範圍 (μm)：單元週期，需滿足 r_i < period/2
MIN_PERIOD = 0.2
MAX_PERIOD = 0.4

# t_r (μm)：pattern 層厚度
T_R_MIN = 0.01
T_R_MAX = 0.3

# t_s (μm)：substrate 層厚度
T_S_MIN = 0.01
T_S_MAX = 0.2

# t_g (μm)：底層 ground 厚度，固定（對吸收影響小）
T_G = 0.2

# =============================================================================
# 主流程：運行 N_SAMPLES 次模擬
# =============================================================================

for i in range(N_SAMPLES):
    print(f"[{i+1}/{N_SAMPLES}]", end=" ")
    resolution = random.choice(RESOLUTION_OPTIONS)
    t_r = round(random.uniform(T_R_MIN, T_R_MAX), 4)
    t_s = round(random.uniform(T_S_MIN, T_S_MAX), 4)
    ring_run_simulation(resolution, MIN_PERIOD, MAX_PERIOD, t_r, t_s, T_G, random_mode=True)
