# robot_arm_sizing_explicit_distances.py
# DIY 6DOF Arm – refined worst-case with explicit distances (a..g)

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g = 9.81  # m/s^2

# -------------------- PARAMS (edit me) --------------------
# Geometrie (m)
L1, L2, L3 = 0.18, 0.18, 0.04

# Distanzen a..g (m)
a = 0.05          # shoulder -> M_phi3 (auf/nahe L1)
b = 0.5 * L1      # shoulder -> CoM(L1)
c = 0.00          # elbow   -> M_phi4
d = 0.03          # elbow   -> M_phi5
e = 0.5 * L2      # elbow   -> CoM(L2)
f = 0.00          # wrist   -> M_phi6
g_ = 0.5 * L3     # wrist   -> CoM(L3)

# Zusätzliche Querradien (m) für Rollachsen (J4/J6) und Massenschwerpunkte
r_L2 = 0.025      # typischer Radius des Unterarm-Segments (L2)
r_M4 = 0.00      # radialer Abstand der Motoren M4/M5 zur Rollachse
r_M5 = 0.020
r_M6 = 0.010      # radialer Abstand Motor M6 zur Rollachse J6
r_tool = 0.025    # effektiver Tool-Radius
r_payload = 0.025 # effektiver Payload-Radius (angenommener Greiferbereich)


# Massen (kg)
m_payload = 0.80
m_tooling = 0.10
tool_frac_on_L3 = 0.5
m_L1, m_L2, m_L3 = 0.250, 0.150, 0.080
m_M3, m_M4, m_M5, m_M6 = 0.692, 0.310, 0.310, 0.140

# Zielkinematik (Default)
omega_out_max_default = np.deg2rad(45.0)  # rad/s
alpha_out_default = 3.0                   # rad/s^2
SF = 1.5

# Gelenk-spezifische Kinematik (bei Bedarf anpassen)
omega_out_max_per_joint = {
    "J1": np.deg2rad(45),
    "J2": np.deg2rad(50),
    "J3": np.deg2rad(45),
    "J4": np.deg2rad(120),
    "J5": np.deg2rad(90),
    "J6": np.deg2rad(180),
}
alpha_out_per_joint = {
    j: alpha_out_default for j in ["J1", "J2", "J3", "J4", "J5", "J6"]
}

# Getriebedaten je Gelenk (Gesamt: Planetary * Riemen)
@dataclass
class Gear:
    ratio: float       # N (Abtrieb/Motordrehwinkel)
    efficiency: float  # 0..1

gears = {
    "J1": Gear(6.4, 0.95),
    "J2": Gear(20, 0.95),
    "J3": Gear(18.0952381, 0.95),
    "J4": Gear(4, 0.95),
    "J5": Gear(4.0, 0.95),
    "J6": Gear(1.0, 1.0),
}

# Motor-Kurven (rpm, Nm) — DEINE Werte bleiben unverändert
MOTOR_CURVES: Dict[str, List[Tuple[float, float]]] = {
    "NEMA17_20mm_16Ncm": [
        (0.0,   0.160), (37.5,  0.064), (75.0,  0.118), (150.0, 0.118),
        (225.0, 0.119), (300.0, 0.118), (375.0, 0.114), (450.0, 0.113),
        (525.0, 0.108), (600.0, 0.098), (675.0, 0.099), (750.0, 0.078),
    ],
    "NEMA17_40mm_45Ncm": [
        (0.0, 0.45), (60.0, 0.22), (90.0, 0.34), (120.0, 0.33), (150.0, 0.325),
        (220.0, 0.32), (300.0, 0.315), (360.0, 0.30), (450.0, 0.29), (520.0, 0.28),
    ],
    "NEMA17_60mm_65Ncm": [
        (0.0, 0.65), (40.0, 0.65), (100.0, 0.64), (150.0, 0.56), (195.0, 0.64),
        (250.0, 0.64), (300.0, 0.55), (345.0, 0.55), (395.0, 0.48), (450.0, 0.39),
        (500.0, 0.39),
    ],
}

# **WICHTIG**: Motor pro Gelenk wählen
motor_per_joint = {
    "J1": "NEMA17_60mm_65Ncm",
    "J2": "NEMA17_60mm_65Ncm",
    "J3": "NEMA17_40mm_45Ncm",
    "J4": "NEMA17_40mm_45Ncm",
    "J5": "NEMA17_40mm_45Ncm",
    "J6": "NEMA17_20mm_16Ncm",
}

# -------------------- helpers --------------------
def interp_torque(curve: List[Tuple[float, float]], rpm: float) -> float:
    xs = np.array([p[0] for p in curve], dtype=float)
    ys = np.array([p[1] for p in curve], dtype=float)
    if rpm <= xs[0]:
        return float(ys[0])
    if rpm >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(rpm, xs, ys))

def rpm(rad_per_s: float) -> float:
    return float(rad_per_s * 60.0 / (2 * np.pi))

FG = lambda m: m * g  # Gewichtskraft

# -------------------- Statik (J2/J3) --------------------
r_J2 = {
    "M6":   L1 + L2 + f,
    "L3":   L1 + L2 + g_,
    "M4":   L1 + c,
    "M5":   L1 + d,
    "L2":   L1 + e,
    "M3":   a,
    "L1":   b,
    "PL":   L1 + L2 + L3,
    "TOOL": L1 + L2 + tool_frac_on_L3 * L3,
}
M_J2_static = (
    r_J2["M6"] * FG(m_M6) + r_J2["L3"] * FG(m_L3) + r_J2["M4"] * FG(m_M4) +
    r_J2["M5"] * FG(m_M5) + r_J2["L2"] * FG(m_L2) + r_J2["M3"] * FG(m_M3) +
    r_J2["L1"] * FG(m_L1) + r_J2["PL"] * FG(m_payload) + r_J2["TOOL"] * FG(m_tooling)
)

r_J3 = {
    "M6":   L2 + f,
    "L3":   L2 + g_,
    "M4":   c,
    "M5":   d,
    "L2":   e,
    "PL":   L2 + L3,
    "TOOL": L2 + tool_frac_on_L3 * L3,
}
M_J3_static = (
    r_J3["M6"] * FG(m_M6) + r_J3["L3"] * FG(m_L3) + r_J3["M4"] * FG(m_M4) +
    r_J3["M5"] * FG(m_M5) + r_J3["L2"] * FG(m_L2) + r_J3["PL"] * FG(m_payload) +
    r_J3["TOOL"] * FG(m_tooling)
)

# -------------------- grobe Dynamik (I = Σ m r^2) --------------------
def I_point_sum_J2() -> float:
    terms = [
        (m_M6, r_J2["M6"]), (m_L3, r_J2["L3"]), (m_M4, r_J2["M4"]), (m_M5, r_J2["M5"]),
        (m_L2, r_J2["L2"]), (m_M3, r_J2["M3"]), (m_L1, r_J2["L1"]), (m_payload, r_J2["PL"]),
        (m_tooling, r_J2["TOOL"]),
    ]
    return float(sum(m * r**2 for m, r in terms))

def I_point_sum_J3() -> float:
    terms = [
        (m_M6, r_J3["M6"]), (m_L3, r_J3["L3"]), (m_M4, r_J3["M4"]), (m_M5, r_J3["M5"]),
        (m_L2, r_J3["L2"]), (m_payload, r_J3["PL"]), (m_tooling, r_J3["TOOL"]),
    ]
    return float(sum(m * r**2 for m, r in terms))

def I_J1() -> float:
    # Trägheit um die Basis-Yaw-Achse: alle Massen mit ihren horizontalen Hebeln
    terms = [
        (m_M6, r_J2["M6"]), (m_L3, r_J2["L3"]), (m_M4, r_J2["M4"]), (m_M5, r_J2["M5"]),
        (m_L2, r_J2["L2"]), (m_M3, r_J2["M3"]), (m_L1, r_J2["L1"]),
        (m_payload, r_J2["PL"]), (m_tooling, r_J2["TOOL"]),
    ]
    return float(sum(m * r**2 for m, r in terms))

def I_J4() -> float:
    # Unterarm-Roll: Polarträgheit um Längsachse
    return (
        0.5 * m_L2 * r_L2**2 +
        m_M4 * r_M4**2 + m_M5 * r_M5**2 +
        m_M6 * r_M6**2 +
        m_tooling * r_tool**2 +
        m_payload * r_payload**2
    )

def I_J5() -> float:
    # Wrist-Pitch: Punktmassenmodell um Handgelenkachse
    return (
        m_L3 * g_**2 +
        m_M6 * f**2 +
        m_tooling * (tool_frac_on_L3 * L3)**2 +
        m_payload * L3**2
    )

def I_J6() -> float:
    # Tool-Roll: Polarträgheit um Tool-Achse
    return (
        m_M6 * r_M6**2 +
        0.5 * m_tooling * r_tool**2 +
        0.5 * m_payload * r_payload**2
    )

I_J2 = I_point_sum_J2()
I_J3 = I_point_sum_J3()

M_J2_dyn = I_J2 * alpha_out_per_joint["J2"]
M_J3_dyn = I_J3 * alpha_out_per_joint["J3"]

# Abtriebsmomente (mit SF)
tau_out_J2 = (M_J2_static + M_J2_dyn) * SF
tau_out_J3 = (M_J3_static + M_J3_dyn) * SF

# -------------------- einfache Modelle für J1/J4/J5/J6 --------------------
def tau_out_required_per_joint() -> Dict[str, float]:
    out = {}

    # J2, J3 aus exakter Rechnung
    out["J2"] = tau_out_J2
    out["J3"] = tau_out_J3

    # J1: Basis-Yaw – nur Dynamik
    out["J1"] = I_J1() * alpha_out_per_joint["J1"] * SF

    # J4: Unterarm-Roll – nur Dynamik
    out["J4"] = I_J4() * alpha_out_per_joint["J4"] * SF

    # J5: Wrist-Pitch – Statik + Dynamik
    tau_J5_static = (
        g_ * FG(m_L3) +
        f   * FG(m_M6) +
        (tool_frac_on_L3 * L3) * FG(m_tooling) +
        L3 * FG(m_payload)
    )
    tau_J5_dyn = I_J5() * alpha_out_per_joint["J5"]
    out["J5"] = (tau_J5_static + tau_J5_dyn) * SF

    # J6: Tool-Roll – nur Dynamik
    out["J6"] = I_J6() * alpha_out_per_joint["J6"] * SF

    return out


# -------------------- Motor-Seite & Bewertung --------------------
def size_joint(joint: str) -> dict:
    tau_out_req = tau_out_required_per_joint()[joint]
    N   = gears[joint].ratio
    eta = gears[joint].efficiency
    omega_out_max_j = omega_out_max_per_joint[joint]

    rpm_motor_max   = rpm(omega_out_max_j * N)
    tau_motor_req   = tau_out_req / (N * eta)

    motor_name      = motor_per_joint[joint]
    curve           = MOTOR_CURVES[motor_name]
    tau_motor_avail = interp_torque(curve, rpm_motor_max)
    margin          = (tau_motor_avail / tau_motor_req - 1.0) * 100.0

    return dict(
        joint=joint,
        motor=motor_name,
        N=N, eta=eta,
        rpm_motor_at_max=rpm_motor_max,
        tau_out_req_Nm=tau_out_req,
        tau_motor_req_Nm=tau_motor_req,
        tau_motor_avail_Nm=tau_motor_avail,
        margin_percent=margin,
    )

# Breakdown-Tabelle Schulter – Transparenz der statischen Anteile
def shoulder_breakdown_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "Term": [
            "(L1+L2+f)*FG(M6)",
            "(L1+L2+g)*FG(L3)",
            "(L1+c)*FG(M4)",
            "(L1+d)*FG(M5)",
            "(L1+e)*FG(L2)",
            "a*FG(M3)",
            "b*FG(L1)",
            "(L1+L2+L3)*FG(Payload)",
            "(L1+L2+tool_frac*L3)*FG(Tool)",
        ],
        "Lever_arm_m": [r_J2["M6"], r_J2["L3"], r_J2["M4"], r_J2["M5"], r_J2["L2"], r_J2["M3"], r_J2["L1"], r_J2["PL"], r_J2["TOOL"]],
        "Force_N":     [FG(m_M6),  FG(m_L3),  FG(m_M4),  FG(m_M5),  FG(m_L2),  FG(m_M3),  FG(m_L1),  FG(m_payload),  FG(m_tooling)],
    })
    df["Moment_Nm"] = df["Lever_arm_m"] * df["Force_N"]
    return df

# -------------------- Output --------------------
def main() -> None:
    joints = ["J1", "J2", "J3", "J4", "J5", "J6"]
    rows = [size_joint(j) for j in joints]
    df = pd.DataFrame(rows)

    print("\n=== Sizing summary (all 6 joints) ===")
    print(df.round(3).to_string(index=False))

    print("\n=== Shoulder static torque breakdown (horizontal) ===")
    print(shoulder_breakdown_df().round(3).to_string(index=False))

    # Plot: Kurven je verwendetem Motor + Arbeitspunkte pro Gelenk
    plt.figure(figsize=(8, 6))
    used_motors = {row["motor"] for row in rows}
    for name in used_motors:
        curve = np.array(MOTOR_CURVES[name])
        plt.plot(curve[:, 0], curve[:, 1], label=f"{name} curve")

    for row in rows:
        plt.scatter([row["rpm_motor_at_max"]], [row["tau_motor_req_Nm"]],
                    s=60, marker="o", label=f'{row["joint"]} req')

    plt.xlabel("Motor speed [rpm]")
    plt.ylabel("Motor torque [Nm]")
    plt.title("Torque–speed check @ max joint speed")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
