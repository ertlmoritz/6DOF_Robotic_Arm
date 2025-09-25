# DIY 6DOF Arm – refined worst-case with explicit distances (a..g)
#
# What’s new vs previous template:
# - Explicit distances for where each mass “acts”:
#   a: shoulder -> M_phi3 (elbow motor on/near L1)
#   b: shoulder -> CoM of L1 (usually 0.5*L1)
#   c: elbow    -> M_phi4 along L2
#   d: elbow    -> M_phi5 along L2
#   e: elbow    -> CoM of L2 (usually 0.5*L2)
#   f: wrist    -> M_phi6 along L3
#   g: wrist    -> CoM of L3 (usually 0.5*L3)
# - Static torque at shoulder (J2) uses your full formula (horizontal pose).
# - Static torque at elbow (J3) uses analogous formula about the elbow.
# - Dynamic inertia (very rough) assembled from point-mass r^2 sums using the same distances.
#
# Edit the PARAMS section and re-run. A summary table + torque-speed plot appear.
#
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caas_jupyter_tools import display_dataframe_to_user
from pathlib import Path

g = 9.81

# -------------------- PARAMS (edit me) --------------------
# Geometry
L1, L2, L3 = 0.16, 0.14, 0.10   # [m]
# Distances (see header). Defaults reflect typical placements.
a = 0.03          # [m] shoulder -> M_phi3 (on L1)
b = 0.5 * L1      # [m] shoulder -> CoM(L1)
c = 0.02          # [m] elbow   -> M_phi4
d = 0.04          # [m] elbow   -> M_phi5
e = 0.5 * L2      # [m] elbow   -> CoM(L2)
f = 0.02          # [m] wrist   -> M_phi6
g_ = 0.5 * L3     # [m] wrist   -> CoM(L3)

# Masses
m_payload = 1.00
m_tooling = 0.30     # assume located at wrist/L3 with an offset below
tool_frac_on_L3 = 0.9  # 0..1 along L3 from wrist towards TCP
m_L1, m_L2, m_L3 = 0.35, 0.25, 0.10
m_M3, m_M4, m_M5, m_M6 = 0.20, 0.18, 0.18, 0.16  # motor masses (examples)

# Desired kinematics
omega_out_max = np.deg2rad(45.0)  # rad/s
alpha_out = 3.0                   # rad/s^2
SF = 1.5

@dataclass
class Gear:
    ratio: float
    efficiency: float

gears = {
    "J1": Gear(60.0, 0.70),
    "J2": Gear(40.0, 0.76),
    "J3": Gear(40.0, 0.76),
}

MOTOR_CURVES: Dict[str, List[Tuple[float, float]]] = {
    "NEMA17_generic": [
        (0.0,   0.55),
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
    if rpm <= xs[0]: return float(ys[0])
    if rpm >= xs[-1]: return float(ys[-1])
    return float(np.interp(rpm, xs, ys))

def rpm(rad_per_s: float) -> float:
    return float(rad_per_s * 60.0 / (2*np.pi))

# -------------------- static torque (horizontal) --------------------
# Forces (weights)
FG = lambda m: m * g

# Shoulder (J2) – your formula:
# M_J2 = (L1+L2+f)*FG(M6) + (L1+L2+g)*FG(L3) + (L1+c)*FG(M4) + (L1+d)*FG(M5)
#      + (L1+e)*FG(L2) + a*FG(M3) + b*FG(L1) + (L1+L2+L3)*FG(payload) + tool term
r_J2 = {
    "M6": L1 + L2 + f,
    "L3": L1 + L2 + g_,
    "M4": L1 + c,
    "M5": L1 + d,
    "L2": L1 + e,
    "M3": a,
    "L1": b,
    "PL": L1 + L2 + L3,
    "TOOL": L1 + L2 + tool_frac_on_L3 * L3
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

# Elbow (J3): analogous about elbow, only distal + elements on L2
# M_J3 = (L2+f)*FG(M6) + (L2+g)*FG(L3) + c*FG(M4) + d*FG(M5) + e*FG(L2)
#      + (L2+L3)*FG(payload) + (L2+tool_frac*L3)*FG(tool)
r_J3 = {
    "M6": L2 + f,
    "L3": L2 + g_,
    "M4": c,
    "M5": d,
    "L2": e,
    "PL": L2 + L3,
    "TOOL": L2 + tool_frac_on_L3 * L3
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

# -------------------- crude dynamics via point-mass inertia --------------------
# I = sum(m * r^2) with same radii as above, for the respective joint axes.
def I_point_sum_J2():
    terms = [
        (m_M6, r_J2["M6"]), (m_L3, r_J2["L3"]), (m_M4, r_J2["M4"]), (m_M5, r_J2["M5"]),
        (m_L2, r_J2["L2"]), (m_M3, r_J2["M3"]), (m_L1, r_J2["L1"]), (m_payload, r_J2["PL"]),
        (m_tooling, r_J2["TOOL"])
    ]
    return sum(m * r**2 for m, r in terms)

def I_point_sum_J3():
    terms = [
        (m_M6, r_J3["M6"]), (m_L3, r_J3["L3"]), (m_M4, r_J3["M4"]), (m_M5, r_J3["M5"]),
        (m_L2, r_J3["L2"]), (m_payload, r_J3["PL"]), (m_tooling, r_J3["TOOL"])
    ]
    return sum(m * r**2 for m, r in terms)

I_J2 = I_point_sum_J2()
I_J3 = I_point_sum_J3()

M_J2_dyn = I_J2 * alpha_out
M_J3_dyn = I_J3 * alpha_out

# Required output torques with safety
tau_J2_req_out = (M_J2_static + M_J2_dyn) * SF
tau_J3_req_out = (M_J3_static + M_J3_dyn) * SF

# -------------------- motor-side requirements --------------------
motor_curve = MOTOR_CURVES[selected_motor]

def motor_requirements(joint: str, tau_out_req: float, omega_out_max: float):
    N = gears[joint].ratio
    eta = gears[joint].efficiency
    tau_motor_req = tau_out_req / (N * eta)
    rpm_motor_max = rpm(omega_out_max * N)
    tau_motor_avail = interp_torque(motor_curve, rpm_motor_max)
    margin = (tau_motor_avail / tau_motor_req - 1.0) * 100.0
    return dict(
        joint=joint, N=N, eta=eta,
        tau_out_req_Nm=tau_out_req,
        rpm_motor_at_max=rpm_motor_max,
        tau_motor_req_Nm=tau_motor_req,
        tau_motor_avail_Nm=tau_motor_avail,
        margin_percent=margin
    )

rows = [
    motor_requirements("J2", tau_J2_req_out, omega_out_max),
    motor_requirements("J3", tau_J3_req_out, omega_out_max),
]

# Yaw J1 (dynamic only, rough): use a guess; refine later if needed.
I_J1_guess = 0.05
tau_J1_req_out = I_J1_guess * alpha_out * SF
rows.append(motor_requirements("J1", tau_J1_req_out, omega_out_max))

df = pd.DataFrame(rows)

# Show a detailed breakdown table for the static shoulder torque terms (so you see contributions)
shoulder_breakdown = pd.DataFrame({
    "Term": ["(L1+L2+f)*FG(M6)", "(L1+L2+g)*FG(L3)", "(L1+c)*FG(M4)", "(L1+d)*FG(M5)",
             "(L1+e)*FG(L2)", "a*FG(M3)", "b*FG(L1)", "(L1+L2+L3)*FG(Payload)", "(L1+L2+t*L3)*FG(Tool)"],
    "Lever_arm_m": [r_J2["M6"], r_J2["L3"], r_J2["M4"], r_J2["M5"], r_J2["L2"], r_J2["M3"], r_J2["L1"], r_J2["PL"], r_J2["TOOL"]],
    "Force_N": [FG(m_M6), FG(m_L3), FG(m_M4), FG(m_M5), FG(m_L2), FG(m_M3), FG(m_L1), FG(m_payload), FG(m_tooling)],
})
shoulder_breakdown["Moment_Nm"] = shoulder_breakdown["Lever_arm_m"] * shoulder_breakdown["Force_N"]

display_dataframe_to_user("Sizing summary (with explicit a..g)", df.round(3))
display_dataframe_to_user("Shoulder static torque breakdown (horizontal pose)", shoulder_breakdown.round(3))

# Plot motor torque-speed curve + operating points
plt.figure(figsize=(7,5))
curve = np.array(MOTOR_CURVES[selected_motor])
plt.plot(curve[:,0], curve[:,1], label=f"{selected_motor} curve")
for r in rows:
    plt.scatter([r["rpm_motor_at_max"]], [r["tau_motor_req_Nm"]], s=60, label=f'{r["joint"]} req')
plt.xlabel("Motor speed [rpm]")
plt.ylabel("Motor torque [Nm]")
plt.title("Torque–speed check @ max joint speed")
plt.legend()
plt.tight_layout()
plt.show()

# Save as a standalone file for download
code_path = Path("/mnt/data/robot_arm_sizing_explicit_distances.py")
# Create a minimal but complete, single-file version (without this notebook's display calls)
standalone = f'''# 6DOF arm sizing with explicit distances a..g (standalone)