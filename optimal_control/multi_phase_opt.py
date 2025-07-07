#!/usr/bin/env python3
# ------------------------------------------------------------
# double_phase_var_dt.py : CasADi demo, 2-phase OCP with free dt
# ------------------------------------------------------------
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --------------------- problem sizes ------------------------
nx, nu = 2, 1     # state=[q, qdot], input=[u] (only phase-1 uses u)
N1, N2  = 40, 60  # number of shooting intervals in each phase

# --------------------- dynamics -----------------------------
def f_active(x, u):
    """Active phase: external torque u plus damping"""
    q, qdot = x[0], x[1]
    qddot   = u[0] - 0.2*qdot          # toy model
    return ca.vcat([qdot, qddot])

def f_passive(x):
    """Passive phase:  damping, u≡0"""
    q, qdot = x[0], x[1]
    qddot   = - 0.2*qdot
    return ca.vcat([qdot, qddot])

# --------------------- opti instance ------------------------
opti = ca.Opti()

# ===== phase-1 variables =====
X1  = opti.variable(nx, N1 + 1)       # states q,qdot
U1  = opti.variable(nu, N1)           # control inputs
dt1 = opti.variable(1,  N1)           # step sizes Δt_k > 0

# ===== phase-2 variables =====
X2  = opti.variable(nx, N2 + 1)
dt2 = opti.variable(1,  N2)           # Δt_k > 0, u ≡ 0 so not declared

# --------------------- boundary conditions ------------------
x_init = np.array([0.0, 0.0])
x_goal = np.array([1.0, 0.0])

opti.subject_to(X1[:, 0]     == x_init)   # initial state
# opti.subject_to(X2[:, -1]    == x_goal)   # terminal state
q_goal = 1.0                           # 只给目标位置
opti.subject_to(X2[0, -1] == q_goal) 
opti.subject_to(X1[:, -1]    == X2[:, 0]) # phase connection

# --------------------- positivity of dt ---------------------
dt_min = 0.01
dt_max = 0.05   
opti.subject_to(dt1<=dt_max)           # 50 ms 视系统带宽调
opti.subject_to(dt2 <= dt_max)
opti.subject_to(dt1 >= dt_min)
opti.subject_to(dt2 >= dt_min)

# --------------------- dynamics constraints -----------------
for k in range(N1):
    x_next = X1[:, k] + dt1[0, k] * f_active(X1[:, k], U1[:, k])
    opti.subject_to(X1[:, k + 1] == x_next)

for k in range(N2):
    x_next = X2[:, k] + dt2[0, k] * f_passive(X2[:, k])
    opti.subject_to(X2[:, k + 1] == x_next)

# --------------------- control bounds (phase-1) -------------
u_min, u_max = 0, 3.0
opti.subject_to(opti.bounded(u_min, U1, u_max))

# --------------------- cost function ------------------------
J  = 0
# ── phase-1: control effort
for k in range(N1):
    J += ca.sumsqr(U1[:, k]) * dt1[0, k]
# ── phase-2: damp out velocity
# for k in range(N2):
#     J += ca.sumsqr(X2[1, k]) * dt2[0, k]
# ── penalise total time (weight γ)
# gamma = 1.0
# J += gamma * (ca.sum1(dt1))

opti.minimize(J)

# --------------------- initial guesses ----------------------
opti.set_initial(X1, ca.repmat(x_init, 1, N1 + 1))
opti.set_initial(X2, ca.repmat(x_goal, 1, N2 + 1))
opti.set_initial(U1,  0)
opti.set_initial(dt1, 0.05)
opti.set_initial(dt2, 0.05)

# --------------------- solver settings ----------------------
opti.solver(
    "ipopt",
    {
        "print_time": False,        # CasADi 自己理解
        "ipopt.print_level": 0      # 传递给 Ipopt
        # 或者 "ipopt": {"print_level": 0}
    }
)

sol = opti.solve()

# --------------------- extract & print ----------------------
x1_opt = sol.value(X1)
x2_opt = sol.value(X2)
u1_opt = sol.value(U1)

dt1_opt = sol.value(dt1)
dt2_opt = sol.value(dt2)
positions= x1_opt[0, :].tolist() + x2_opt[0, :].tolist()
velocities = x1_opt[1, :].tolist() + x2_opt[1, :].tolist()
# print(velocities)

gap = sol.value(X1[:, -1] - X2[:, 0])
print("phase-gap =", gap)  
plt.figure(figsize=(10, 5))
plt.plot(positions, label="Position q", marker='o')
plt.plot(velocities, label="Velocity qdot", marker='x')
plt.xlabel("Time step")
plt.ylabel("Position q")
plt.title("Optimized  over Time")
plt.grid()
plt.legend()
# plt.show()
plt.savefig("optimized_pos_vel.png")


print(f"Total phase-1 time  : {dt1_opt.sum():.3f} s")
print(f"Total phase-2 time  : {dt2_opt.sum():.3f} s")
print(f"Objective value J   : {sol.value(J):.4f}")
print(x1_opt)
print(x2_opt)
print(u1_opt)
print(dt1_opt)
print(dt2_opt)