import copy


# class StateVariables:
# 	def __init__():
# 		self.positions: List[float, float, float]
# 		self.velocities: List[float, float, float]


# class States:
# 	def __init__():
# 		self._time_to_state: Dict[int, StateVariables]

# 	def get_state(time: int) -> StateVariables:
# 		return copy.deepcopy(self._time_to_state[time])

# 	def add_state(time: int, state: StateVariables) -> None:
# 		self._time_to_state[time] = state

# 	def get_positions(time: int) -> float:
# 		state = self.get_state(time)
# 		return state.position


class States:
	position_x_row = 0
	position_y_row = 1
	position_z_row = 2
	velocity_x_row = 3
	velocity_y_row = 4
	velocity_z_row = 5
	def __init__():
		self._states: NDArray

	def get_time_col(time: int):
		return time

	def get_state(time: int) -> NDArray:
		column = self.get_time_col(time)
		return self._states[:, column]

	def get_position_y(time: int) -> float:
		column = self.get_time_col(time)
		return self._states[self.position_y_row, column] 


def rk4(f, x, dx, t, dt) -> None:
	''' Performs Runge-Kutta 4 integration method 

	:param f: RHS of the differential equation (dx/dt = f(x,t))
	:param x: Vector to store the approximated solution to x(t)
	:param dx: Vector to store the value of f(x,t) at each time step (it is useful info)
	:param t: Vector storing all timesteps
	:param dt: Time step interval in seconds
	'''

	for i in range(1, len(t)):

		# x[:, i-1] -> time step t
		# x[:, i] 	-> time step t + dt

		xi = x[:, i-1]

		ti = t[i-1]

		rk4_step(xi. ti)

		f1 = f(xi, ti)
		k1 = dt*f1

		f2 = f(xi + 0.5*k1, ti + 0.5*dt)
		k2 = dt*f2

		f3 = f(xi + 0.5*k2, ti + 0.5*dt)
		k3 = dt*f3

		f4 = f(xi + k3, ti + dt)
		k4 = dt*f4

		x[:, i]  = xi + ( k1 + 2*k2 + 2*k3 + k4 )/6
		dx[:, i] = f1

def rk4_step(f, x, dx, dt):
	''' Performs Runge-Kutta 4 integration method for 1 time-step (real time simulation)

	:param f: RHS of the differential equation (dx/dt = f(x,t))
	:param x: Vector to store the approximated solution to x(t)
	:param dx: Vector to store the value of f(x,t)
	:param dt: Time step interval in seconds
	'''

	f1 = f(x, 0)
	k1 = dt * f1

	f2 = f(x + 0.5 * k1, 0)
	k2 = dt * f2

	f3 = f(x + 0.5 * k2, 0)
	k3 = dt * f3

	f4 = f(x + k3, 0)
	k4 = dt * f4

	x[:] = x + (k1 + 2*k2 + 2*k3 + k4)/6
	dx[:] = f1
