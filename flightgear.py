import time
from flightgear_python.fg_if import FDMConnection
from flightsym import makeStateEq
from core import rk4_step
from utilities import loadModel, loadInitialCondition
import numpy as np
import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("Nenhum joystick detectado")

joystick = pygame.joystick.Joystick(0)
joystick.init()


PRINT_EACH = 30
frame = 0
control_type = "PS4"

def control_input():

    pygame.event.pump()

    ail = joystick.get_axis(2)   
    ele = -joystick.get_axis(1)   
    rud = 0.0
    throttle = (joystick.get_axis(5) + 1.0)/2.0
    brake = (joystick.get_axis(4) + 1.0)/2.0
    if control_type == "XBOX":
        if joystick.get_button(4):  # Left Bumper
            rud -= 1.0
        if joystick.get_button(5):  # Right Bumper
            rud += 1.0
    else:  # PS4
        if joystick.get_button(9):  # L1
            rud -= 1.0
        if joystick.get_button(10):  # R1
            rud += 1.0

    return ele, ail, rud, throttle, brake


model = loadModel("aircraftModel.dat")
x = loadInitialCondition("initialCondition.dat")

dx = x*0

f = makeStateEq(model, control_input)

dt = 1.0 / 240.0

R_EARTH = 6378137.0


lat0_deg = 19.7205
lon0_deg = -155.0610


lat0 = np.radians(lat0_deg)
lon0 = np.radians(lon0_deg)

def ned_to_geodetic(xE, yE):
    
    d_lat = xE / R_EARTH
    d_lon = yE / (R_EARTH * np.cos(lat0))

    lat = lat0 + d_lat
    lon = lon0 + d_lon

    return lat, lon

SEND_HZ = 60
FDM_HZ = 240
send_interval = FDM_HZ // SEND_HZ  # 240 / 60 = 4
frame = 0

def fdm_callback(fdm_data, event_pipe):
    global x, dx, frame

    # Integra múltiplos passos internos para cada callback
    for _ in range(send_interval):
        rk4_step(f, x, dx, dt)

    # Extrai posição, orientação e altitude
    xE, yE, zE = x[0], x[1], x[2]
    h = -zE
    phi, theta, psi = x[3], x[4], x[5]
    lat, lon = ned_to_geodetic(xE, yE)

    # Atualiza dados do FlightGear
    fdm_data.lon_rad = lon
    fdm_data.lat_rad = lat
    fdm_data.alt_m = h
    fdm_data.phi_rad = phi
    fdm_data.theta_rad = theta
    fdm_data.psi_rad = psi

    if (frame % PRINT_EACH == 0):
        ele, ail, rud, throttle, brake = control_input()
        V = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)
        u,v,w = x[6], x[7], x[8]
        Vair = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / Vair)
        print(f"ail: {ail:.2f}, ele: {ele:.2f}, rud: {rud:.2f}, throttle: {throttle:.2f}, brake: {brake:.2f}, V: {V:.2f}, alt: {h:.0f}, alpha: {np.degrees(alpha):.1f}°, beta: {np.degrees(beta):.1f}°")

    frame += 1
    return fdm_data
     
if __name__ == "__main__":

    fdm_conn = FDMConnection()

    fdm_conn.connect_rx('localhost', 5501, fdm_callback)

    fdm_conn.connect_tx('localhost',5502)

    fdm_conn.start()

    while True:
        time.sleep(1)
