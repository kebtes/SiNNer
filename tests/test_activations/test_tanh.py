import unittest
import numpy as np
from nnf.activations.tanh import Tanh

class TestTanh(unittest.TestCase):
    
    def test_forward(self):
        tanh = Tanh()

        # Test with a simple input (2D array)
        inputs = np.array([[0.0, 1.0, -1.0],
                           [2.0, -2.0, 0.5]])
        
        # Compute the tanh output
        output = tanh.forward(inputs)
        
        # Ensure the output is between -1 and 1
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
        
    def test_backward(self):
        tanh = Tanh()

        # Test with a simple input (2D array)
        inputs = np.array([[0.0, 1.0, -1.0],
                           [2.0, -2.0, 0.5]])

        # Compute the tanh output (forward pass)
        tanh.forward(inputs)
        
        # Now create a simple gradient (dvalues) coming from the loss function
        dvalues = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]])

        # Perform the backward pass
        dinputs = tanh.backward(dvalues)

        # Ensure that the backward pass returns the correct shape
        self.assertEqual(dinputs.shape, dvalues.shape)
        
        # Ensure the values returned from backward are correct
        expected_dinputs = dvalues * (1 - tanh.output ** 2)
        np.testing.assert_array_equal(dinputs, expected_dinputs)

    def test_forward_numerical_stability(self):
        tanh = Tanh()

        # Test with inputs having large values (for numerical stability)
        inputs = np.array([[1000, 1000, 1000],
                           [-1000, -1000, -1000]])

        # Compute the tanh output
        output = tanh.forward(inputs)

        # Check if output is a valid value between -1 and 1
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
        
    def test_invalid_input(self):
        tanh = Tanh()

        # Test with invalid input (e.g., NaNs or Infs)
        inputs = np.array([[np.nan, 1.0, 2.0],
                           [1.0, np.inf, 3.0]])

        # Expect forward pass to raise an exception due to invalid values
        with self.assertRaises(ValueError):
            tanh.forward(inputs)


if __name__ == '__main__':
    unittest.main()
