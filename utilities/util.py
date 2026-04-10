from typing import Dict

import numpy as np


def loadModel(path: str) -> Dict[str, float]:
    expected_keys = {
        "mass", KEY_INERTIA_X, "Iy", "Iz", "Ixz", "S", "b", "c"
    }

    model = {}

    with open(path, "r") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "#" in line:                     
                line = line.split("#", 1)[0].strip()

            if "=" not in line:
                raise ValueError(f"Line {lineno}: invalid format")

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key not in expected_keys:
                raise ValueError(f"Line {lineno}: invalid key '{key}'")

            try:
                model[key] = float(value)
            except ValueError:
                raise ValueError(f"Line {lineno}: invalid value for '{key}'")

    missing = expected_keys - model.keys()
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    return model


def reportModel(model):

    print("--- INERTIA ---")
    print(f"Mass: {model['mass']}")
    print(f"Ix: {model['Ix']}")
    print(f"Iy: {model['Iy']}")
    print(f"Iz: {model['Iz']}")
    print(f"Ixz: {model['Ixz']}")

    print("\n--- GEOMETRY---")
    print(f"S: {model['S']}")
    print(f"b: {model['b']}")
    print(f"c: {model['c']}")

def loadInitialCondition(path):

    expected_keys = {
        "x0","y0","z0",
        "phi0","tet0","psi0",
        "u0","v0","w0",
        "p0","q0","r0"
    }

    data = {}

    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "#" in line:                     
                line = line.split("#", 1)[0].strip()

            if "=" not in line:
                raise ValueError(f"Line {lineno}: Invalid format")

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            try:
                data[key] = float(value)
            except ValueError:
                raise ValueError(f"Line {lineno}: Invalid value for '{key}'")

    missing = expected_keys - data.keys()
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    p0 = np.deg2rad(data["p0"])
    q0 = np.deg2rad(data["q0"])
    r0 = np.deg2rad(data["r0"])

    x0 = np.array([
        data["x0"], data["y0"], data["z0"],
        data["phi0"], data["tet0"], data["psi0"],
        data["u0"], data["v0"], data["w0"],
        p0, q0, r0
    ])

    return x0
