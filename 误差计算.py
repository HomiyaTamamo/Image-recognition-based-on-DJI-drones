import tkinter as tk
from tkinter import messagebox
import math

# ==========================================
# WGS84椭球参数
# ==========================================

a = 6378137.0                 # 长半轴
f = 1 / 298.257223563         # 扁率
b = (1 - f) * a               # 短半轴


# ==========================================
# Vincenty逆解算法
# 返回距离（米）
# ==========================================

def vincenty_distance(lat1, lon1, lat2, lon2):

    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)

    L = math.radians(lon2 - lon1)

    U1 = math.atan((1 - f) * math.tan(φ1))
    U2 = math.atan((1 - f) * math.tan(φ2))

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)

    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    λ = L

    for _ in range(200):

        sinλ = math.sin(λ)
        cosλ = math.cos(λ)

        sinσ = math.sqrt(
            (cosU2 * sinλ) ** 2 +
            (cosU1 * sinU2 -
             sinU1 * cosU2 * cosλ) ** 2
        )

        if sinσ == 0:
            return 0.0

        cosσ = (
            sinU1 * sinU2 +
            cosU1 * cosU2 * cosλ
        )

        σ = math.atan2(
            sinσ,
            cosσ
        )

        sinα = (
            cosU1 * cosU2 * sinλ
            / sinσ
        )

        cos2α = 1 - sinα ** 2

        if cos2α == 0:

            cos2σm = 0

        else:

            cos2σm = (
                cosσ -
                2 * sinU1 * sinU2 / cos2α
            )

        C = (
            f / 16
            * cos2α
            * (4 + f * (4 - 3 * cos2α))
        )

        λ_prev = λ

        λ = (
            L +
            (1 - C)
            * f
            * sinα
            * (
                σ +
                C * sinσ *
                (
                    cos2σm +
                    C * cosσ *
                    (
                        -1 +
                        2 * cos2σm ** 2
                    )
                )
            )
        )

        if abs(λ - λ_prev) < 1e-12:
            break

    u2 = (
        cos2α *
        (a * a - b * b)
        / (b * b)
    )

    A = (
        1 +
        u2 / 16384 *
        (
            4096 +
            u2 *
            (
                -768 +
                u2 *
                (
                    320 -
                    175 * u2
                )
            )
        )
    )

    B = (
        u2 / 1024 *
        (
            256 +
            u2 *
            (
                -128 +
                u2 *
                (
                    74 -
                    47 * u2
                )
            )
        )
    )

    Δσ = (
        B *
        sinσ *
        (
            cos2σm +
            B / 4 *
            (
                cosσ *
                (
                    -1 +
                    2 * cos2σm ** 2
                )
                -
                B / 6 *
                cos2σm *
                (
                    -3 +
                    4 * sinσ ** 2
                )
                *
                (
                    -3 +
                    4 * cos2σm ** 2
                )
            )
        )
    )

    s = b * A * (σ - Δσ)

    return s


# ==========================================
# 计算按钮
# ==========================================

def calculate():

    try:

        lat1 = float(entry_lat1.get())
        lon1 = float(entry_lon1.get())

        lat2 = float(entry_lat2.get())
        lon2 = float(entry_lon2.get())

        distance = vincenty_distance(
            lat1,
            lon1,
            lat2,
            lon2
        )-5

        result_var.set(
            f"定位误差：{distance:.3f} 米"
        )

    except ValueError:

        messagebox.showerror(
            "输入错误",
            "请输入合法的小数格式经纬度！"
        )

    except Exception as e:

        messagebox.showerror(
            "计算失败",
            str(e)
        )


# ==========================================
# GUI
# ==========================================

root = tk.Tk()

root.title("WGS84高精度定位误差计算")

root.geometry("550x380")

root.resizable(True, True)

# 标题
title_label = tk.Label(
    root,
    text="WGS84高精度定位误差计算系统",
    font=("微软雅黑", 18)
)

title_label.pack(pady=15)

# 输入区域
frame = tk.Frame(root)

frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)

frame.columnconfigure(1, weight=1)

# 第一组

tk.Label(
    frame,
    text="第一组纬度:"
).grid(
    row=0,
    column=0,
    sticky="w",
    pady=5
)

entry_lat1 = tk.Entry(
    frame,
    font=("微软雅黑", 12)
)

entry_lat1.grid(
    row=0,
    column=1,
    sticky="ew",
    pady=5
)

tk.Label(
    frame,
    text="第一组经度:"
).grid(
    row=1,
    column=0,
    sticky="w",
    pady=5
)

entry_lon1 = tk.Entry(
    frame,
    font=("微软雅黑", 12)
)

entry_lon1.grid(
    row=1,
    column=1,
    sticky="ew",
    pady=5
)

# 第二组

tk.Label(
    frame,
    text="第二组纬度:"
).grid(
    row=2,
    column=0,
    sticky="w",
    pady=5
)

entry_lat2 = tk.Entry(
    frame,
    font=("微软雅黑", 12)
)

entry_lat2.grid(
    row=2,
    column=1,
    sticky="ew",
    pady=5
)

tk.Label(
    frame,
    text="第二组经度:"
).grid(
    row=3,
    column=0,
    sticky="w",
    pady=5
)

entry_lon2 = tk.Entry(
    frame,
    font=("微软雅黑", 12)
)

entry_lon2.grid(
    row=3,
    column=1,
    sticky="ew",
    pady=5
)

# 按钮

calc_button = tk.Button(
    root,
    text="计算定位误差",
    font=("微软雅黑", 13),
    height=2,
    command=calculate
)

calc_button.pack(
    fill="x",
    padx=40,
    pady=15
)

# 结果显示

result_var = tk.StringVar()

result_var.set("定位误差：")

result_label = tk.Label(
    root,
    textvariable=result_var,
    font=("微软雅黑", 16),
    fg="blue"
)

result_label.pack(pady=10)

# 说明

tip_label = tk.Label(
    root,
    text="算法：Vincenty (WGS84椭球模型)\n精度可达毫米级",
    fg="gray"
)

tip_label.pack()

root.mainloop()