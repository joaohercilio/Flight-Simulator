from enum import Enum, auto
import json



class Model:
	"""
	Parameters for model of plane usin format from program X.

	"""
	def __init__(self, inertia_x: float, inertia_y: float):
		"""
		inertia_x: Inertial in direction X
		"""
		self.assert_valid_inertia(inertia_x)
		self.assert_valid_inertia(inertia_y)

		self.inertia_x = inertia_x  # Inertia in x direction
		self.inertia_y = inertia_y

	def assert_valid_inertia(value: float) -> None:
		if value <= 0:
			raise ValueError(f"Inertial must be positive, got {value}")

	@classmethod
	def read_file(cls, path: str) -> 'Model':
		# do stuff
		model = ?
		return model

	def write_eclipse_file(self):
		# write .dat with parameters
		pass



model = Model(1.0, 2.0)

model.create_memento()

model2 = Model.load_memento(filename)

