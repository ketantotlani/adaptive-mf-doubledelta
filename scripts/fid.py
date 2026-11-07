"""
Cheap vs accurate lift on one airfoil: how the gap between two fidelities grows with angle of attack.

  Cheap    : 2D source-vortex panel method (Hess-Smith), inviscid, 200 panels, runs in milliseconds.
  Reference: NeuralFoil v0.3.3 (Sharpe & Hansman 2025, https://github.com/peterdsharpe/NeuralFoil),
             viscous, Re = 3e6. Used here as a stand-in for RANS.
  Airfoil  : NACA 2412, angles -4 to 20 deg.

Illustration only: one 2D airfoil, not the double-delta wing benchmark.

Requirements: pip install numpy matplotlib neuralfoil==0.3.3
Run:          python fid.py   -> prints a table and writes fid.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuralfoil as nf


def naca4(m, p, t, n=100):
    """NACA 4-digit airfoil, cosine spacing, ordered TE (lower) -> LE -> TE (upper)."""
    b = np.linspace(0, np.pi, n + 1)
    x = (1 - np.cos(b)) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    X = np.concatenate([xl[::-1], xu[1:]])
    Y = np.concatenate([yl[::-1], yu[1:]])
    X[0] = X[-1] = 1.0
    Y[0] = Y[-1] = 0.0
    return X, Y


def panel_cl(X, Y, alpha_deg):
    """Lift coefficient from a Hess-Smith panel method (constant sources + one vortex strength, Kutta condition)."""
    alpha = np.radians(alpha_deg)
    N = len(X) - 1
    xm, ym = (X[:-1] + X[1:]) / 2, (Y[:-1] + Y[1:]) / 2
    dx, dy = X[1:] - X[:-1], Y[1:] - Y[:-1]
    L, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    # influence of each panel j at each control point i (unit source / unit vortex), global frame
    px, py = xm[:, None], ym[:, None]
    xl = (px - X[None, :-1]) * np.cos(th) + (py - Y[None, :-1]) * np.sin(th)
    yl = -(px - X[None, :-1]) * np.sin(th) + (py - Y[None, :-1]) * np.cos(th)
    r1, r2 = np.hypot(xl, yl), np.hypot(xl - L, yl)
    beta = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    beta = np.where(beta > np.pi, beta - 2 * np.pi, beta)
    beta = np.where(beta < -np.pi, beta + 2 * np.pi, beta)
    us_l, vs_l = np.log(r1 / r2) / (2 * np.pi), beta / (2 * np.pi)
    uv_l, vv_l = vs_l, -us_l
    c, s = np.cos(th), np.sin(th)
    us, vs = us_l * c - vs_l * s, us_l * s + vs_l * c
    uv, vv = uv_l * c - vv_l * s, uv_l * s + vv_l * c
    for i in range(N):  # self-influence
        us[i, i], vs[i, i] = -0.5 * np.sin(th[i]), 0.5 * np.cos(th[i])
        uv[i, i], vv[i, i] = 0.5 * np.cos(th[i]), 0.5 * np.sin(th[i])
    nx, ny, tx, ty = -np.sin(th), np.cos(th), np.cos(th), np.sin(th)
    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)
    A[:N, :N] = us * nx[:, None] + vs * ny[:, None]          # no flow through each panel
    A[:N, N] = (uv * nx[:, None] + vv * ny[:, None]).sum(1)
    rhs[:N] = -(np.cos(alpha) * nx + np.sin(alpha) * ny)
    for k in (0, N - 1):                                       # Kutta condition at the trailing edge
        A[N, :N] += us[k] * tx[k] + vs[k] * ty[k]
        A[N, N] += (uv[k] * tx[k] + vv[k] * ty[k]).sum()
        rhs[N] -= np.cos(alpha) * tx[k] + np.sin(alpha) * ty[k]
    gamma = np.linalg.solve(A, rhs)[N]
    return 2 * gamma * L.sum()                                 # Kutta-Joukowski, chord = 1


X, Y = naca4(0.02, 0.4, 0.12, 100)                             # NACA 2412, 200 panels
al = np.arange(-4, 20.1, 1.0)
cl_panel = np.array([panel_cl(X, Y, a) for a in al])
r = nf.get_aero_from_coordinates(coordinates=np.c_[X[::-1], Y[::-1]], alpha=al, Re=3e6, model_size="large")
cl_nf = np.array(r["CL"]).ravel()

print("alpha  CL_panel  CL_NeuralFoil  |gap|")
for a, b, c in zip(al, cl_panel, cl_nf):
    print(f"{a:5.0f}  {b:8.3f}  {c:13.3f}  {abs(b - c):5.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), dpi=110)
ax[0].plot(al, cl_panel, c="#eb6834", lw=2, label="panel method (inviscid, milliseconds)")
ax[0].plot(al, cl_nf, c="#2a78d6", lw=2, label="NeuralFoil (viscous, Re = 3×10⁶)")
ax[0].fill_between(al, cl_panel, cl_nf, color="#e34948", alpha=0.12)
ax[0].set(xlabel="angle of attack α (deg)", ylabel="lift coefficient C_L", title="Two fidelities, same airfoil: lift")
ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(al, np.abs(cl_panel - cl_nf), color="#e34948", width=0.7)
ax[1].set(xlabel="angle of attack α (deg)", ylabel="|ΔC_L| between the two", title="Their disagreement is small at low α, large near stall")
fig.tight_layout()
fig.savefig("fid.png", facecolor="white")
