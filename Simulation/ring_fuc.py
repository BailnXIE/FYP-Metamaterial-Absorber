

import os
import time
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import csv
import random
# # from meep.materials import SiO2
# from meep.materials import Ni



def ring_run_simulation(current_resolution, min_period, max_period, t_r, t_s, t_g, random_mode=True, radii_group=None):
    # =============================================================================
    # 可自定義參數設置
    # =============================================================================


    nickel = mp.Medium(
        epsilon = 1.1,
        E_susceptibilities=[
            mp.LorentzianSusceptibility(frequency = 3.9858250867132115,gamma = 2.511879899524022,sigma = 4.113838124625053),
            mp.LorentzianSusceptibility(frequency = 1.205259585418557,gamma = 2.1791021647942617,sigma = 33.8282537957956),
            mp.LorentzianSusceptibility(frequency = 5.473956202134901,gamma = 0.0,sigma = 0.6422654532224172),
            mp.DrudeSusceptibility(frequency = 1.0,gamma = 71.40722029347084,sigma = 4.2029657093411785),
            mp.LorentzianSusceptibility(frequency = 0.5391704583285774,gamma = 0.5710356956819119,sigma = 34.3101795523754),
            mp.LorentzianSusceptibility(frequency = 0.5162622607242847,gamma = 3.1517824141461036e-07, sigma = 5.087204485682729e-06),
            mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 26.927018244146122),
        ],
    )
    SiO2 = mp.Medium(
        epsilon = 1.4734,
        E_susceptibilities=[
            mp.DrudeSusceptibility(frequency = 1.0,gamma = 0.0,sigma = 0.00025388935570505225),
            mp.LorentzianSusceptibility(frequency = 35.389765360298895,gamma = 7.127379294672169,sigma = 0.0),
            mp.LorentzianSusceptibility(frequency = 3.4770163700805936,gamma = 0.6710934044071608,sigma = 0.0004169766288133064),
            mp.LorentzianSusceptibility(frequency = 9.066558637023224,gamma = 0.0,sigma = 0.26650430221274624),
            mp.LorentzianSusceptibility(frequency = 9.06655863618851,gamma = 0.0,sigma = 0.4053568509089618),
        ]
    )
    Nickel = nickel
    Air = mp.Medium(index=1.0)
    pec = mp.perfect_electric_conductor
    substrate_material = SiO2
    ground_material = Nickel
    pattern_material = ground_material
    # 空間分辨率
    resolution = current_resolution  # pixels/μm

    # 光源強度
    source_amplitude = 10.0  # 光源振幅（可調整，例如 2.0 表示兩倍強度）

    # 週期邊界範圍 (Y和Z方向的週期)
    period = random.uniform(min_period, max_period)  # 最小半径0.01避免太小
    period = round(period, 3)  # 保留小数点后三位
    period_y = period  # Y方向週期 (μm)
    period_z = period_y  # Z方向週期 (μm)

    # Monitor 位置設置
    monitor_reflection_x = None  # 反射monitor的X位置，None表示自動計算
    monitor_transmission_x = None  # 透射monitor的X位置，None表示自動計算
    monitor_offset = 0.3  # Monitor距離PML的偏移量 (μm)

    # 波長範圍
    min_wavelength = 0.3  # 最小波長 (μm)
    max_wavelength = 2.0  # 最大波長 (μm)
    n_wavelengths = 200  # 波長點數
    # 指定波長：額外繪製一張電場分布圖；設為 None 則不繪製
    custom_efield_wavelength = 1.45  # (μm)，需在 [min_wavelength, max_wavelength] 內

    #幾何高度計算
    Height_geometry = t_r + t_s + t_g
    pattern_height = t_r  
    substrate_thickness = t_s 

    # PML 厚度
    dpml = 0.6*max_wavelength  # PML厚度 (μm)
    dpad = 0.8 * max_wavelength   # 填充層厚度 (μm)
    dgap = Height_geometry * 1.1 

    # =============================================================================
    # 計算域尺寸計算
    # =============================================================================

    # X方向總長度：PML + 填充 + 幾何 + 填充 + PML
    sx = 2 * (dpml + dpad + dgap)
    sy = period_y  # Y方向長度 (單個單元)
    sz = period_z  # Z方向長度 (單個單元)

    cell_size = mp.Vector3(sx, sy, sz)

    # =============================================================================
    # 幾何結構定義函數
    # =============================================================================
    def base_geometry(spacer_hight, ground_height, pattern_height):
        objs = []
        #spcaer
        objs.append(
            mp.Block(
                material=substrate_material,
                size=mp.Vector3(spacer_hight, sy, sz),
                center=mp.Vector3(-spacer_hight/2 - pattern_height/2, 0, 0),
            )
        )
        #ground
        objs.append(
            mp.Block(
                material=ground_material,
                size=mp.Vector3(ground_height, sy, sz),
                center=mp.Vector3(-ground_height/2 - spacer_hight - pattern_height/2, 0, 0),
            )
        )
        return objs

   

    def disk_geometry(radius,material, pattern_height):
        objs = []
        objs.append(
            mp.Cylinder(
                radius=radius,          # (必填) 圓柱的半徑
                height=pattern_height,          # (必填) 圓柱的長度/高度
                axis=mp.Vector3(1,0,0),# (選填) 圓柱朝向，預設是 Z 軸 (0,0,1)
                center=mp.Vector3(0,0,0), # (選填) 中心點位置，預設是 (0,0,0)
                material=material         # (必填) 材料
            )
        )
        return objs
  
    def ring_geometry():
        objs = []
        # 从1-10中随机抽取整数n
        n = random.randint(1, 10)
        
        # 在1/2period的范围内随机生成n个radius参数
        max_radius = period / 2.0
        radii = []
        for _ in range(n):
            radius = random.uniform(0.01, max_radius)  # 最小半径0.01避免太小
            radius = round(radius, 3)  # 保留小数点后三位
            radii.append(radius)
        
        # 对radius进行从大到小排序
        radii.sort(reverse=True)
        
        # 逐个生成disk，材料交替：最大的disk是Nickel，第二大是Air，然后Nickel，Air交替
        for i, radius in enumerate(radii):
            if i % 2 == 0:  # 偶数索引（0, 2, 4...）使用Nickel
                material = Nickel
            else:  # 奇数索引（1, 3, 5...）使用Air
                material = Air
            
            # 使用disk_geometry生成disk
            disk_objs = disk_geometry(radius, material, pattern_height)
            objs.extend(disk_objs)
        
        # 返回几何对象、半径列表和半径数量
        return objs, radii, n


    def create_geometry():
        geometry = []
        geometry.extend(base_geometry(t_s, t_g, t_r))
        if random_mode:
            ring_objs, radii, n_rings = ring_geometry()
        else:
            if radii_group is None or len(radii_group) == 0:
                raise ValueError("random_mode=False 時必須提供非空的 radii_group")
            radii = sorted([round(float(r), 3) for r in radii_group], reverse=True)
            n_rings = len(radii)
            ring_objs = []
            for i, radius in enumerate(radii):
                material = Nickel if i % 2 == 0 else Air
                ring_objs.extend(disk_geometry(radius, material, pattern_height))
        geometry.extend(ring_objs)
        return geometry, radii, n_rings

    # =============================================================================
    # 光源設置
    # =============================================================================

    # 光源位置：在填充層中間
    source_x = (dgap) + (0.75 * dpad)
    source_position = mp.Vector3(source_x, 0, 0)

    # 中心波長和頻率
    lcen = (min_wavelength + max_wavelength) / 2
    fcen = 1.0 / lcen
    df = 2.0 * (1.0 / min_wavelength - 1.0 / max_wavelength)

    # 高斯脈衝光源
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
            component=mp.Ey,  # Ey極化
            center=source_position,
            size=mp.Vector3(0, sy, sz),  # 平面光源，覆蓋整個Y-Z平面
            amplitude=source_amplitude,  # 光源強度
        )
    ]

    # =============================================================================
    # 邊界條件設置
    # =============================================================================

    # PML層設置 (只在X方向，Y和Z使用週期性邊界)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

    # 週期性邊界條件 (Y和Z方向週期性)
    k_point = mp.Vector3(0, 0, 0)

    # =============================================================================
    # 創建模擬對象
    # =============================================================================

    geometry, radii, n_rings = create_geometry()

    # 計算monitor位置 (如果未指定)
    if monitor_reflection_x is None:
        monitor_reflection_x = (dgap) + (0.25 * dpad)
    if monitor_transmission_x is None:
        # Transmission monitor 放在空氣間隙中間（表面背面和PML之間）
        monitor_transmission_x = -(dgap) - (0.25 * dpad)

    # 波長和頻率數組
    wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)
    frequencies = 1.0 / wavelengths

    # =============================================================================
    # 參考模擬：無結構，獲取真正的入射強度
    # =============================================================================

    print("進行參考模擬（無結構）以獲取入射強度...")

    # 創建參考模擬（無結構）
    sim_ref = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=[],  # 無結構 (空)
        k_point=k_point,
        sources=sources,
        default_material=mp.Medium(index=1.0),  # 默認材料為空氣
        # eps_averaging=True,
    )

    # 在參考模擬中添加透射監測器（用於獲取入射強度）
    flux_ref_reflection = sim_ref.add_flux(
        frequencies,
        mp.FluxRegion(
            center=mp.Vector3(monitor_reflection_x, 0, 0), # 注意：這裡必須是 reflection 的位置
            size=mp.Vector3(0, sy, sz),
        )
    )

    # 運行參考模擬
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(0, 0, 0), 1e-4
    ))
    # run_time = sx + 50 
    # sim_ref.run(until=run_time)

    # 獲取參考入射強度（無結構時的透射強度即為入射強度）
    incident_refl_field_data = sim_ref.get_flux_data(flux_ref_reflection)
    incident_intensity = np.array(mp.get_fluxes(flux_ref_reflection))
    print(f"參考入射強度範圍: {incident_intensity.min():.6e} - {incident_intensity.max():.6e}")

    # 重置參考模擬
    sim_ref.reset_meep()

    # =============================================================================
    # 主模擬：有結構
    # =============================================================================

    print("\n進行主模擬（有結構）...")

    # 創建主模擬（有結構）
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        k_point=k_point,
        sources=sources,
        default_material=mp.Medium(index=1.0),  # 默認材料為空氣
        eps_averaging=True,
        Courant = 0.2,
    )

    # =============================================================================
    # 添加Flux Monitors
    # =============================================================================

    print("設置Flux Monitors...")

    # 反射光監測器 (結構上方)
    flux_reflection = sim.add_flux(
        frequencies,
        mp.FluxRegion(
            center=mp.Vector3(monitor_reflection_x, 0, 0),
            size=mp.Vector3(0, sy, sz),  # 平面monitor
        )
    )

    # 透射光監測器 (結構下方，在空氣間隙中)
    flux_transmission = sim.add_flux(
        frequencies,
        mp.FluxRegion(
            center=mp.Vector3(monitor_transmission_x, 0, 0),
            size=mp.Vector3(0, sy, sz),  # 平面monitor
        )
    )

    # =============================================================================
    # 添加Mode Monitor用於S參數計算
    # =============================================================================
    # Mode monitor用於計算S11 (反射)
    mode_monitor_reflection = sim.add_mode_monitor(
        frequencies,
        mp.ModeRegion(
            center=mp.Vector3(monitor_reflection_x, 0, 0),
            size=mp.Vector3(0, sy, sz),
            direction=mp.X,
        ),
    )

    # Mode monitor用於計算S21 (透射)
    mode_monitor_transmission = sim.add_mode_monitor(
        frequencies,
        mp.ModeRegion(
            center=mp.Vector3(monitor_transmission_x, 0, 0),
            size=mp.Vector3(0, sy, sz),
            direction=mp.X,
        ),
    )

    # =============================================================================
    # 添加DFT場監控器 (用於電場分布)
    # =============================================================================

    # 選擇幾個關鍵頻率進行電場監控
    dft_freqs = [
        1.0 / min_wavelength,
        fcen,
        1.0 / max_wavelength,
    ]
    if custom_efield_wavelength is not None and min_wavelength <= custom_efield_wavelength <= max_wavelength:
        dft_freqs.append(1.0 / custom_efield_wavelength)

    dft_fields = sim.add_dft_fields(
        [mp.Ey],  # 捕獲Ey電場分量
        dft_freqs,
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(sx, sy, sz),  # 監控幾何結構區域
    )

    print(f"Monitor設置完成:")
    print(f"- 反射監測器: x={monitor_reflection_x:.3f} μm")
    print(f"- 透射監測器: x={monitor_transmission_x:.3f} μm (在空氣間隙中)")
    print(f"- 監控頻率: {len(frequencies)}個頻率點")
    print(f"- 使用參考模擬的入射強度進行歸一化")

    # =============================================================================
    # 運行模擬
    # =============================================================================

    print("\n開始運行模擬...")
    print("=" * 50)

    # 在運行模擬前提示反射監視器需要減去入射數據
    sim.load_minus_flux_data(flux_reflection, incident_refl_field_data)

    # 運行模擬直到場衰減
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(0, 0, 0), 1e-4
    ))

    print("模擬完成!")

    # =============================================================================
    # 獲取Flux數據
    # =============================================================================

    print("\n分析結果...")

    # 獲取flux數據
    reflection_flux = np.array(mp.get_fluxes(flux_reflection))
    transmission_flux = np.array(mp.get_fluxes(flux_transmission))

    # 使用參考模擬的入射強度進行歸一化
    # 計算TRA
    T = np.abs(transmission_flux / incident_intensity)  # 透射率
    R = np.abs(reflection_flux / incident_intensity)  # 反射率
    A = 1 - R - T  # 吸收率

    # 確保A在合理範圍內
    A = np.clip(A, 0, 1)

    # =============================================================================
    # 找出吸收峰值
    # =============================================================================

    # 找出所有局部極大值
    if len(A) > 2:
        candidate_idx = np.where((A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]))[0] + 1
        if candidate_idx.size > 0:
            # 找出最大峰值
            peak_idx = candidate_idx[np.argmax(A[candidate_idx])]
            # 找出第二高峰
            if candidate_idx.size > 1:
                # 排除第一高峰，找出第二高峰
                candidate_values = A[candidate_idx]
                sorted_indices = np.argsort(candidate_values)[::-1]  # 降序排列
                second_peak_idx = candidate_idx[sorted_indices[1]]
                second_peak_wavelength = wavelengths[second_peak_idx]
                second_peak_absorption = A[second_peak_idx]
            else:
                second_peak_idx = None
                second_peak_wavelength = None
                second_peak_absorption = None
        else:
            peak_idx = np.argmax(A)
            second_peak_idx = None
            second_peak_wavelength = None
            second_peak_absorption = None
    else:
        peak_idx = np.argmax(A)
        second_peak_idx = None
        second_peak_wavelength = None
        second_peak_absorption = None

    peak_wavelength = wavelengths[peak_idx]
    peak_absorption = A[peak_idx]

    print(f"\n吸收峰值:")
    print(f"- 第一高峰波長: {peak_wavelength:.4f} μm")
    print(f"- 第一高峰吸收率: {peak_absorption*100:.2f}%")
    if second_peak_idx is not None:
        print(f"- 第二高峰波長: {second_peak_wavelength:.4f} μm")
        print(f"- 第二高峰吸收率: {second_peak_absorption*100:.2f}%")
    else:
        print(f"- 未找到第二高峰")

    # =============================================================================
    # 繪製TRA曲線 (標示吸收峰值)
    # =============================================================================

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, T, label="Transmission", color="green", linewidth=2)
    plt.plot(wavelengths, R, label="Reflection", color="blue", linewidth=2)
    plt.plot(wavelengths, A, label="Absorption", color="red", linewidth=2)

    # 我們把峰值的具體數值直接寫在 label 裡，這樣它就會出現在圖例框中
    peak_label = f"Max Peak: {peak_wavelength:.3f} μm ({peak_absorption*100:.1f}%)"
    plt.scatter(peak_wavelength, peak_absorption, color="black", s=100, zorder=5, label=peak_label)

    # loc='best' 會自動找空白位置，或者您可以指定 'upper right'
    # framealpha=0.9 讓框框背景稍微不透明一點，避免擋住線看不清
    # edgecolor='gray' 給框框加個邊
    plt.legend(
        loc="best", 
        fontsize=11, 
        frameon=True, 
        fancybox=True, 
        framealpha=0.9, 
        edgecolor="gray",
        # title="Simulation Results" # 可選：給框框加個標題
    )
    plt.xlabel("Wavelength (μm)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("TRA vs Wavelength (Meta Absorber)", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.xlim(min_wavelength, max_wavelength)
    plt.tight_layout()
    #保存
    output_folder = "./data_graph"
    # 確保文件夾存在，如果不存就自動建立一個
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已建立新文件夾: {output_folder}")
    # 2. 設定基礎檔名
    base_name = "TRA_beehive"
    file_ext = ".png"
    final_path = os.path.join(output_folder, base_name + file_ext)
    # 3. 檢查重名並自動改名 (核心邏輯)
    counter = 1
    while os.path.exists(final_path):
        # 如果檔案已經存在，就加上 _1, _2, _3...
        new_name = f"{base_name}_{counter}{file_ext}"
        final_path = os.path.join(output_folder, new_name)
        counter += 1
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    print(f"TRA曲線圖已保存為: {final_path}")
    plt.close()

    # =============================================================================
    # 繪製YZ截面圖
    # =============================================================================

    # YZ結構截面圖 (Y=0平面，顯示結構和監測器)
    sim.plot2D(
        output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, sy, sz)),
    )
    plt.title("YZ Cross-section After Simulation (Structure)", fontsize=12, fontweight="bold")
    plt.xlabel("Y (μm)", fontsize=11)
    plt.ylabel("Z (μm)", fontsize=11)
    plt.tight_layout()
    output_folder = "./data_graph"
    # 確保文件夾存在，如果不存就自動建立一個
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已建立新文件夾: {output_folder}")
    # 2. 設定基礎檔名
    base_name = "YZ_cross_section_beehive"
    file_ext = ".png"
    final_path = os.path.join(output_folder, base_name + file_ext)
    # 3. 檢查重名並自動改名 (核心邏輯)
    counter = 1
    while os.path.exists(final_path):
        # 如果檔案已經存在，就加上 _1, _2, _3...
        new_name = f"{base_name}_{counter}{file_ext}"
        final_path = os.path.join(output_folder, new_name)
        counter += 1
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"YZ截面圖已保存為 {final_path}")




    # =============================================================================
    # 繪製吸收峰值時的電場分布圖
    # =============================================================================

    print("\n開始繪製電場分佈圖 (Auto-Reshape Mode)...")

    try:
        # 1. 找到最接近峰值波長的DFT頻率索引
        peak_freq = 1.0 / peak_wavelength
        dft_freq_array = np.array(dft_freqs)
        freq_idx = int(np.argmin(np.abs(dft_freq_array - peak_freq)))

        # 2. 獲取原始數據
        # get_dft_array 返回複數場 (Complex)
        Ey_data_full = sim.get_dft_array(dft_fields, mp.Ey, freq_idx)
        # get_epsilon 返回實數 (Real)
        eps_data_full = sim.get_epsilon()

        # 定義一個輔助函數來安全地獲取截面並修正維度
        # 保存外部的peak_wavelength變量，避免參數名衝突
        outer_peak_wavelength = peak_wavelength
        def plot_cross_section(slice_Ey, slice_eps, axis1_name, axis2_name, axis1_range, axis2_range, filename, xlim_range=None, peak_wl=None):
            """
            slice_Ey: 電場切片 (2D array)
            slice_eps: 結構切片 (2D array)
            axis1_range: (min, max) for 第一維
            axis2_range: (min, max) for 第二維
            peak_wl: 峰值波長，用於標題顯示
            """
            # 1. 獲取電場數據的形狀
            n1, n2 = slice_Ey.shape
            
            # 2. 處理 Epsilon 尺寸不匹配問題 (裁剪或填充)
            # 如果 eps 比 Ey 大，裁剪 eps
            if slice_eps.shape[0] > n1: slice_eps = slice_eps[:n1, :]
            if slice_eps.shape[1] > n2: slice_eps = slice_eps[:, :n2]
            # 如果 eps 比 Ey 小 (極少見)，裁剪 Ey 以匹配 eps (取交集)
            if slice_eps.shape[0] < n1: 
                slice_Ey = slice_Ey[:slice_eps.shape[0], :]
                n1 = slice_Ey.shape[0]
            if slice_eps.shape[1] < n2: 
                slice_Ey = slice_Ey[:, :slice_eps.shape[1]]
                n2 = slice_Ey.shape[1]

            # 3. 根據最終的形狀生成坐標
            ax1 = np.linspace(axis1_range[0], axis1_range[1], n1)
            ax2 = np.linspace(axis2_range[0], axis2_range[1], n2)
            X_grid, Y_grid = np.meshgrid(ax1, ax2, indexing='ij')

            # 4. 繪圖
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 畫電場 (Amplitude)
            # 注意：使用 np.abs 取幅度
            pcm = ax.contourf(X_grid, Y_grid, np.abs(slice_Ey), levels=100, cmap='inferno')
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label('|Ey| Amplitude (V/m)', fontsize=12, rotation=270, labelpad=20)
            
            # 畫結構輪廓 (Epsilon)
            ax.contour(X_grid, Y_grid, slice_eps, levels=[1.1], colors='white', linewidths=1.0, alpha=0.7)

            ax.set_xlabel(f"{axis1_name} (μm)", fontsize=12)
            ax.set_ylabel(f"{axis2_name} (μm)", fontsize=12)
            # 使用傳入的peak_wl參數，如果沒有則使用外部的peak_wavelength
            wavelength_to_use = peak_wl if peak_wl is not None else outer_peak_wavelength
            ax.set_title(f"{axis1_name}{axis2_name} Plane @ λ={wavelength_to_use:.3f} μm", fontsize=14, fontweight="bold")

            if xlim_range is not None:
                 ax.set_xlim(xlim_range)
            
            # 保存
            output_folder = "./data_graph"
            if not os.path.exists(output_folder): os.makedirs(output_folder)
            
            save_path = os.path.join(output_folder, filename)
            # 簡單防覆蓋
            if os.path.exists(save_path):
                save_path = save_path.replace(".png", f"_{int(time.time())}.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> 圖表已保存: {save_path}")

        # ==========================================
        # 執行繪圖 (傳入數據切片)
        # ==========================================
        
        # 數據全尺寸
        nx_full, ny_full, nz_full = Ey_data_full.shape

        my_x_limit = (-dgap, 0.2 * dgap)

        # --- YZ 截面 (X=Center) ---
        # 幾何裡面的圓柱 / pattern 都是放在原點 x=0
        # DFT 區域也是以 (0,0,0) 為中心、長度為 sx
        # 因此「穿過結構中心的 YZ 平面」就是整個 DFT 陣列的 X 中心那一層
        x_idx = nx_full // 2

        # 繪製第一高峰的電場截面圖
        plot_cross_section(
            Ey_data_full[x_idx, :, :],
            eps_data_full[x_idx, :, :],
            "Y", "Z", (-sy/2, sy/2), (-sz/2, sz/2),
            f"YZ_Ey_peak_{peak_wavelength:.3f}um.png",
            xlim_range=None,
            peak_wl=peak_wavelength
        )

        # 繪製第二高峰的電場截面圖（如果存在）
        if second_peak_idx is not None:
            # 找到最接近第二高峰波長的DFT頻率索引
            second_peak_freq = 1.0 / second_peak_wavelength
            second_freq_idx = int(np.argmin(np.abs(dft_freq_array - second_peak_freq)))
            
            # 獲取第二高峰的電場數據
            Ey_data_second = sim.get_dft_array(dft_fields, mp.Ey, second_freq_idx)
            
            # 繪製第二高峰的電場截面圖
            plot_cross_section(
                Ey_data_second[x_idx, :, :],
                eps_data_full[x_idx, :, :],
                "Y", "Z", (-sy/2, sy/2), (-sz/2, sz/2),
                f"YZ_Ey_second_peak_{second_peak_wavelength:.3f}um.png",
                xlim_range=None,
                peak_wl=second_peak_wavelength
            )
            print(f"第二高峰電場圖已繪製完成！")

        # 繪製指定波長的電場截面圖（若已設置且落在波長範圍內）
        if custom_efield_wavelength is not None and min_wavelength <= custom_efield_wavelength <= max_wavelength:
            custom_freq = 1.0 / custom_efield_wavelength
            custom_freq_idx = int(np.argmin(np.abs(dft_freq_array - custom_freq)))
            Ey_data_custom = sim.get_dft_array(dft_fields, mp.Ey, custom_freq_idx)
            plot_cross_section(
                Ey_data_custom[x_idx, :, :],
                eps_data_full[x_idx, :, :],
                "Y", "Z", (-sy/2, sy/2), (-sz/2, sz/2),
                f"YZ_Ey_custom_{custom_efield_wavelength:.3f}um.png",
                xlim_range=None,
                peak_wl=custom_efield_wavelength
            )
            print(f"指定波長 {custom_efield_wavelength:.3f} μm 電場圖已繪製完成！")

        print("所有電場圖繪製完成！")

    except Exception as e:
        print(f"繪圖過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # 計算S參數
    # =============================================================================

    print("\n計算S參數...")

    # 獲取本徵模式係數
    eig_parity = mp.NO_PARITY
    eps = 1e-12 # 防止除以零的極小值

    # 初始化數組 (全部填入 nan，以防計算失敗)
    S11_mag = np.full(len(wavelengths), np.nan)
    S11_phase = np.full(len(wavelengths), np.nan)

    try:
        # S11 (反射係數)
        mode_data_ref = sim.get_eigenmode_coefficients(
            mode_monitor_reflection, [1], eig_parity=eig_parity
        )
        
        if mode_data_ref is not None:
            # 在反射 monitor 位置：
            # - 前向模式（forward）是入射波（在有結構情況下的入射波）
            # - 後向模式（backward）是反射波
            # S11 = 反射波 / 入射波（都在同一個位置測量）
            forward_coeff_ref = mode_data_ref.alpha[0, :, 0]  # 前向模式（入射波）
            backward_coeff_ref = mode_data_ref.alpha[0, :, 1]  # 後向模式（反射波）
            
            # 相位從本征模式係數計算：使用同一個 monitor 位置的前向模式作為參考
            S11_complex = np.where(
                np.abs(forward_coeff_ref) > eps,
                backward_coeff_ref / forward_coeff_ref,
                np.nan
            )
            S11_phase = np.unwrap(np.angle(S11_complex))
            
            # 幅度從功率反射率計算：|S11| = sqrt(R)
            # 確保 R 在 [0, 1] 範圍內（被動元件限制）
            R_clipped = np.clip(R, 0, 1)
            S11_mag = np.sqrt(R_clipped)
        else:
            S11_mag = np.full(len(wavelengths), np.nan)
            S11_phase = np.full(len(wavelengths), np.nan)
            print("警告: 無法從反射monitor獲取S11數據")
    except Exception as e:
        print(f"計算S11時發生錯誤: {e}")

    try:
        # S21 (透射係數)
        mode_data_tran = sim.get_eigenmode_coefficients(
            mode_monitor_transmission, [1], eig_parity=eig_parity
        )
        
        if mode_data_tran is not None:
            # 幅度從功率透射率計算：|S21| = sqrt(T)
            # 確保 T 在 [0, 1] 範圍內（被動元件限制）
            T_clipped = np.clip(T, 0, 1)
            S21_mag = np.sqrt(T_clipped)
        else:
            S21_mag = np.full(len(wavelengths), np.nan)
            print("警告: 無法從透射monitor獲取S21數據")
    except Exception as e:
        print(f"計算S21時發生錯誤: {e}")
        S21_mag = np.full(len(wavelengths), np.nan)

    # =============================================================================
    # 保存CSV文件
    # =============================================================================

    # =============================================================================
    # 保存 Absorption 數據 (CSV)
    # =============================================================================
    # 1. 建立專門存放數據的文件夾
    data_folder = "./data"

    # 如果文件夾不存在，就建立它
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # 2. 使用固定的文件名
    filename = "ring_data.csv"
    csv_file_path = os.path.join(data_folder, filename)

    # 3. 數據整合與檢查
    # 再次確保 A 在 0~1 之間
    A = np.clip(A, 0, 1)

    # 4. 構建CSV格式：第一行是labels，第二行是數值
    # Labels: [resolution, period, n_rings, r_1, r_2, ..., r_10, t_r, t_s, t_g, Abs(0.30um), Abs(0.31um), ..., Abs(2.00um)]
    labels = ["resolution", "period", "n_rings"]
    
    # 添加半径labels（最多10个）
    max_radius_labels = 10
    for i in range(1, max_radius_labels + 1):
        labels.append(f"r_{i}")
    
    # 添加其他参数
    labels.extend(["t_r", "t_s", "t_g"])
    
    # 添加Absorption的labels，格式為 Abs(波長um)
    abs_labels = [f"Abs({w:.2f}um)" for w in wavelengths]
    labels.extend(abs_labels)
    
    # 構建數值行：參數值 + Absorption值
    values = [resolution, period_y, n_rings]
    
    # 添加半径值（最多10个，不足的用空字符串填充）
    for i in range(max_radius_labels):
        if i < len(radii):
            values.append(radii[i])
        else:
            values.append("")  # 空值表示没有这个半径
    
    # 添加其他参数
    values.extend([t_r, t_s, t_g])
    
    # 添加Absorption數組
    values.extend(A.tolist())
    
    # 5. 檢查文件是否存在，決定是創建新文件還是追加數據
    file_exists = os.path.exists(csv_file_path)
    
    if file_exists:
        # 文件已存在，追加模式：只寫入數據行（不寫header）
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(values)
        print(f"📝 數據已追加到現有文件: {csv_file_path}")
    else:
        # 文件不存在，創建新文件：寫入header和第一行數據
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 寫入labels行
            writer.writerow(labels)
            # 寫入數值行
            writer.writerow(values)
        print(f"📄 已創建新文件並寫入數據: {csv_file_path}")

    print("-" * 40)
    print(f"✅ Absorption 數據已成功保存！")
    print(f"📂 文件路徑: {csv_file_path}")
    print(f"📊 包含參數: period={period_y}, n_rings={n_rings}, radii={radii}, t_r={t_r}, t_s={t_s}, t_g={t_g}")
    print(f"📈 Absorption數據點數: {len(A)}")
    print("-" * 40)

    # =============================================================================
    # 輸出總結
    # =============================================================================

    print("\n" + "=" * 50)
    print("模擬完成！所有結果已保存。")
    print(f"  波長: {peak_wavelength:.4f} μm")
    print(f"  吸收率: {peak_absorption*100:.2f}%")
    print(f"  分辨率: {resolution}%")
    print("=" * 50)

    sim.reset_meep()


if __name__ == "__main__":
    resolution = 201
    min_period = 0.28  # 最小周期
    max_period = 0.32  # 最大周期     
    radii_group = [0.141,0.140,0.095,0.085,0.067,0.058,0.031,0.024,0.020]
    t_r=0.092
    t_s=0.091
    t_g=0.195
    period=0.298
    ring_run_simulation(resolution, period, period, t_r, t_s, t_g,
                        random_mode=False, radii_group=radii_group)

    # 第二次：隨機模式
    # ring_run_simulation(resolution, min_period, max_period, t_r, t_s, t_g,
    #                     random_mode=True)
