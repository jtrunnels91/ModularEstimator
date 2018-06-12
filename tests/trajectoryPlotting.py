from context import modest as md
import numpy as np
from pulsarData.loadPulsarData import loadPulsarData

from pypet import Environment, cartesian_product, Trajectory

from pint import UnitRegistry
ureg = UnitRegistry()

def generateRandomNumbers(traj):
    traj.f_add_result(
        'value.$',
        np.random.normal(traj.loc, traj.std, traj.size),
        comment='Random numbers'
    )
    return

env = Environment(
    filename='./plotTestResults',
    trajectory='randomNumberGeneration',
    add_time=True
)

traj=env.trajectory

traj.f_add_parameter('loc', np.float64(1))
traj.f_add_parameter('std', np.float64(1), comment='Standard deviation of random number distribution')
traj.f_add_parameter('size', np.float64(1), comment='Number of random numbers per data point to generate')

traj.f_explore(
    cartesian_product(
        {
            'std': np.logspace(-2,4,7),
            'size': np.ones(10)
        }
    )
)

env.run(generateRandomNumbers)

plotResultsKeys={
    'results std': {
        'varName': 'value',
        'function': np.std,
        'name': 'results error std',
        'plot': 'line'
        },
    'results scatter': {
        'varName': 'value',
        'function': np.abs,
        'name': 'result scatter',
        'plot': 'scatter'
        }
    }

md.plots.montecarloplots.plotTrajectory(traj, 'std', plotResultsKeys)
