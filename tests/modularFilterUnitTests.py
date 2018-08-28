import unittest
from context import modest as md
import numpy as np

class TestModularFilters(unittest.TestCase):
    def testAddStates(self):
        # Create a simple state class to add to the filter
        class MySimpleState():
            def __init__(self, length, covarianceVal):
                self.length = length
                self.covarianceVal = covarianceVal
                return
            def dimension(self):
                return self.length
            def covariance(self):
                return np.eye(self.length) * self.covarianceVal
            
        stateLength1 = np.random.randint(0,10)
        cov1 = np.random.randint(0,10)
        state1 = MySimpleState(stateLength1, cov1)
        
        stateLength2 = np.random.randint(0,10)
        cov2 = np.random.randint(0,10)
        state2 = MySimpleState(stateLength2, cov2)

        myFilter = md.ModularFilter()
        
        myFilter.addStates('state1', state1)
        myFilter.addStates('state2', state2)

        self.assertEqual(stateLength1 + stateLength2, myFilter.totalDimension)

        from scipy.linalg import block_diag
        stackedCov = block_diag(np.eye(stateLength1) * cov1, np.eye(stateLength2) * cov2)
        self.assertTrue(np.all(stackedCov == myFilter.covarianceMatrix))

unittest.main(argv=['ignored','-v'], exit=False)
