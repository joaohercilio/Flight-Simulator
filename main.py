from utilities import loadModel, loadInitialCondition, generatePlots
from flightsym import runSimulation


def main(model_file: str = None, initial_conditions_file: str) -> None:
    if model_file is None:
        model_file = "aircraftModel.dat"

    model = loadModel(model_file)
    
    x0 = loadInitialCondition(initial_conditions_file)

    t0 = 0
    tf = 10
    dt = 0.01

    t, x, dx = runSimulation(model, x0, t0, tf, dt)

    generatePlots(t, x, dx)


if __name__ == "__main__":
    # model_file = "aircraftModel.dat"
    # initial_conditions_file = "initialCondition.dat"
    # main()

    t, x, dx = runSimulation(model, x0, t0, tf, dt)
