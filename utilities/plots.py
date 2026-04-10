import matplotlib.pyplot as plt
import numpy as np
import os

def clean(x, tol=1e-6):
    x = np.asarray(x)
    y = x.copy()
    y[np.abs(y) < tol] = 0.0
    return y

def wrapAngle(angle):
	return np.arctan2(np.sin(angle), np.cos(angle))

def _loadBlocks(path):
    blocks = []
    current = []

    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "#" in line:                     
                line = line.split("#", 1)[0].strip()

            if not line:
                continue

            if line.upper() == "PLOT":
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line)

    if current:
        blocks.append(current)

    return blocks

def generatePlots(t, x, dx, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots.dat")

    xE, yE, zE = x[0], x[1], x[2]
    h = -zE

    phi, tet, psi = x[3], x[4], x[5]
    p, q, r = x[9], x[10], x[11]

    xdot, ydot, zdot = dx[0], dx[1], dx[2]
    phidot, tetdot, psidot = dx[3], dx[4], dx[5]
    u, v, w = x[6], x[7], x[8]
    V = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / V)

    groups = {
        "Position": [("North [m]", clean(xE)),
                     ("East [m]", clean(yE)),
                     ("Altitude [m]", clean(h))],

        "Velocity NED": [("Velocity North [m/s]", clean(xdot)),
                         ("Velocity East [m/s]", clean(ydot)),
                         ("Velocity Down [m/s]", clean(zdot))],

        "Euler angles": [("phi [deg]", phi*180/np.pi),
                         ("theta [deg]", tet*180/np.pi),
                         ("psi [deg]", wrapAngle(psi)*180/np.pi)],

        "Euler rates": [("phidot [deg/s]", phidot*180/np.pi),
                        ("thetadot [deg/s]", tetdot*180/np.pi),
                        ("psidot [deg/s]", psidot*180/np.pi)],

        "Angular velocity": [("p [deg/s]", p*180/np.pi),
                             ("q [deg/s]", q*180/np.pi),
                             ("r [deg/s]", r*180/np.pi)],

        "Aerodynamics": [("alpha [deg]", alpha*180/np.pi),
                             ("beta [deg]", beta*180/np.pi),
                             ("Airspeed [m/s]", clean(V))]

    }

    blocks = _loadBlocks(config_path)

    for block in blocks:

        n_groups = len(block)
        fig, axs = plt.subplots(n_groups, 3, figsize=(12, 3*n_groups))

        if n_groups == 1:
            axs = np.array([axs])

        for row, group_name in enumerate(block):
            if group_name not in groups:
                continue

            variables = groups[group_name]

            for col in range(3):
                ax = axs[row, col]
                if col < len(variables):
                    label, data = variables[col]
                    ax.plot(t, data)
                    ax.set_ylabel(label)
                    ax.set_xlabel("Time [s]")
                    ax.grid(True)
                else:
                    ax.axis("off")

        plt.tight_layout()

    plt.show()
