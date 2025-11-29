# DIY 6-DOF Robot Arm – Worst-Case Sizing

This repository contains a single Python script for performing rough worst-case sizing of a DIY 6-DOF robotic arm using NEMA17 stepper motors. The script computes required output torques for all joints (static + simplified dynamic), maps them to the motor shaft using gearbox ratio and efficiency, and compares them with predefined torque–speed motor curves.

---

## Features

* Explicit geometry with segment lengths `L1–L3` and distances `a..g`
* Static torque computation for shoulder (`J2`) and elbow (`J3`)
* Simplified inertia models for all joints (`J1–J6`)
* Gearbox parameters per joint (ratio, efficiency)
* Built-in torque–speed curves for multiple NEMA17 motors
* Console output:

  * Sizing summary (required vs. available motor torque + safety margin)
  * Static torque breakdown for shoulder
* Matplotlib visualization of motor curves and operating points

---

## Requirements

```bash
python >= 3.9
pip install numpy pandas matplotlib
```

---

## Usage

1. Clone or download the repository.
2. Adjust geometry, masses, gearbox parameters, and motor assignments at the top of the script.
3. Run:

   ```bash
   python robot_arm_sizing_explicit_distances.py
   ```
4. The script prints:

   * Sizing table for all joints
   * Static shoulder torque breakdown

   A plot with torque–speed curves and joint operating points opens automatically.

---

## Notes

This model provides *engineering-level estimation* only. For detailed design (FEM, compliance, friction, full dynamics), further refinement or additional tools are recommended.
