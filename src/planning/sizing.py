# robot_arm_sizing_explicit_distances.py
# DIY 6DOF Arm – refined worst-case with explicit distances (a..g)
# - Explizite Hebelarme a..g (Motor-/CoM-Abstände)
# - Statik bei horizontalem Worst-Case für J2 (Schulter) & J3 (Ellbogen)
# - Grobe Dynamik: I = Σ m r^2 (mit denselben Hebelarmen)
# - Mapping auf Motorseite (Drehmoment/Speed) inkl. Margin ggü. Motor-Kurve
# - Konsolen-Tabellen + Matplotlib-Plot (optional Save)

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g = 9.81  # m/s^2

# -------------------- PARAMS (edit me) --------------------
# Geometrie (m)
L1, L2, L3 = 0.16, 0.14, 0.10

# Distanzen a..g (m)
a = 0.03          # shoulder -> M_phi3 (auf/nahe L1)
b = 0.5 * L1      # shoulder -> CoM(L1)
c = 0.02          # elbow   -> M_phi4
d = 0.04          # elbow   -> M_phi5
e = 0.5 * L2      # elbow   -> CoM(L2)
f = 0.02          # wrist   -> M_phi6
g_ = 0.5 * L3     # wrist   -> CoM(L3)

# Massen (kg)
m_payload = 1.00
m_tooling = 0.30          # Werkzeug, am L3 (siehe tool_frac_on_L3)
tool_frac_on_L3 = 0.9     # Position des Tool-CoM entlang L3 (0..1) ab Handgelenk
m_L1, m_L2, m_L3 = 0.35, 0.25, 0.10
m_M3, m_M4, m_M5, m_M6 = 0.20, 0.18, 0.18, 0.16  # Motor-/Getriebe-Massen (Beispiel)

# Zielkinematik
omega_out_max = np.deg2rad(45.0)  # rad/s (max Gelenkgeschw.)
alpha_out = 3.0                   # rad/s^2 (Ziel-Beschleunigung)
SF = 1.5                          # Sicherheitsfaktor gesamt (einfacher Start)

# Getriebedaten je Gelenk (Gesamt: Planetary * Riemen)
@dataclass
class Gear:
    ratio: float       # N (Abtrieb/Motordrehwinkel)
    efficiency: float  # 0..1

gears = {
    "J1": Gear(60.0, 0.70),
    "J2": Gear(40.0, 0.76),
    "J3": Gear(40.0, 0.76),
}

# Motor-Torque-Speed-Kurve (rpm, Nm). Punkte aus Datenblatt übernehmen!
MOTOR_CURVES: Dict[str, List[Tuple[float, float]]] = {
    "NEMA17_generic": [
        (0.0,   0.55),   # Holding torque (informativ)
        (100.0, 0.45),
        (300.0, 0.30),
        (600.0, 0.18),
        (900.0, 0.10),
    ],
}
selected_motor = "NEMA17_generic"

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

# -------------------- Statik (horizontal) --------------------
FG = lambda m: m * g  # Gewichtskraft

# Schultermoment J2 (nach deiner Formel, inkl. Tool)
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
    r_J2["M6"] * FG(m_M6) +
    r_J2["L3"] * FG(m_L3) +
    r_J2["M4"] * FG(m_M4) +
    r_J2["M5"] * FG(m_M5) +
    r_J2["L2"] * FG(m_L2) +
    r_J2["M3"] * FG(m_M3) +
    r_J2["L1"] * FG(m_L1) +
    r_J2["PL"] * FG(m_payload) +
    r_J2["TOOL"] * FG(m_tooling)
)

# Ellbogenmoment J3 (analog um Ellbogen; nur distale Beiträge + L2-Anteile)
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
    r_J3["M6"] * FG(m_M6) +
    r_J3["L3"] * FG(m_L3) +
    r_J3["M4"] * FG(m_M4) +
    r_J3["M5"] * FG(m_M5) +
    r_J3["L2"] * FG(m_L2) +
    r_J3["PL"] * FG(m_payload) +
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

I_J2 = I_point_sum_J2()
I_J3 = I_point_sum_J3()

M_J2_dyn = I_J2 * alpha_out
M_J3_dyn = I_J3 * alpha_out

# Erforderliche Abtriebsmomente inkl. Sicherheit
tau_J2_req_out = (M_J2_static + M_J2_dyn) * SF
tau_J3_req_out = (M_J3_static + M_J3_dyn) * SF

# -------------------- Motor-Seite --------------------
motor_curve = MOTOR_CURVES[selected_motor]

def motor_requirements(joint: str, tau_out_req: float, omega_out_max: float) -> dict:
    N   = gears[joint].ratio
    eta = gears[joint].efficiency
    tau_motor_req   = tau_out_req / (N * eta)
    rpm_motor_max   = rpm(omega_out_max * N)
    tau_motor_avail = interp_torque(motor_curve, rpm_motor_max)
    margin          = (tau_motor_avail / tau_motor_req - 1.0) * 100.0
    return dict(
        joint=joint, N=N, eta=eta,
        tau_out_req_Nm=tau_out_req,
        rpm_motor_at_max=rpm_motor_max,
        tau_motor_req_Nm=tau_motor_req,
        tau_motor_avail_Nm=tau_motor_avail,
        margin_percent=margin,
    )

rows = [
    motor_requirements("J2", tau_J2_req_out, omega_out_max),
    motor_requirements("J3", tau_J3_req_out, omega_out_max),
]

# Yaw J1 (nur grobe Dynamikannahme)
I_J1_guess = 0.05
tau_J1_req_out = I_J1_guess * alpha_out * SF
rows.append(motor_requirements("J1", tau_J1_req_out, omega_out_max))

df = pd.DataFrame(rows)

# Breakdown-Tabelle Schulter – für Transparenz
shoulder_breakdown = pd.DataFrame({
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
shoulder_breakdown["Moment_Nm"] = shoulder_breakdown["Lever_arm_m"] * shoulder_breakdown["Force_N"]

# -------------------- Output --------------------
def main() -> None:
    # Tabellen in Konsole ausgeben
    print("\n=== Sizing summary (explicit a..g) ===")
    print(df.round(3).to_string(index=False))

    print("\n=== Shoulder static torque breakdown (horizontal) ===")
    print(shoulder_breakdown.round(3).to_string(index=False))

    # Plot (optional: statt zeigen eine Datei speichern)
    curve = np.array(MOTOR_CURVES[selected_motor])
    plt.figure(figsize=(7, 5))
    plt.plot(curve[:, 0], curve[:, 1], label=f"{selected_motor} curve")
    for r in rows:
        plt.scatter([r["rpm_motor_at_max"]], [r["tau_motor_req_Nm"]], s=60, label=f'{r["joint"]} req')
    plt.xlabel("Motor speed [rpm]")
    plt.ylabel("Motor torque [Nm]")
    plt.title("Torque–speed check @ max joint speed")
    plt.legend()
    plt.tight_layout()

    # Anzeige:
    plt.show()

    # Optional speichern:
    # plt.savefig("torque_speed.png", dpi=150)
    # df.round(3).to_csv("sizing_summary.csv", index=False)
    # shoulder_breakdown.round(3).to_csv("shoulder_breakdown.csv", index=False)

if __name__ == "__main__":
    main()
