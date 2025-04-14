import unittest
import numpy as np
from nnf.activations.softmax import Softmax


class TestSoftmax(unittest.TestCase):
    
    def test_forward(self):
        softmax = Softmax()

        # Test with a simple input (2D array, where each row is a vector of logits)
        inputs = np.array([[1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0]])
        
        # Compute the softmax output
        output = softmax.forward(inputs)
        
        # Ensure the output is a probability distribution (values between 0 and 1)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        
        # Ensure that the sum of probabilities for each row equals 1
        row_sums = np.sum(output, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_backward(self):
        softmax = Softmax()

        # Test with a simple input (2D array, where each row is a vector of logits)
        inputs = np.array([[1.0, 2.0, 3.0],
                           [1.0, 2.0, 3.0]])

        # Compute the softmax output (forward pass)
        softmax.forward(inputs)
        
        # Now create a simple gradient (dvalues) coming from the loss function
        dvalues = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]])

        # Perform the backward pass
        dinputs = softmax.backward(dvalues)

        # Ensure that the backward pass returns the correct shape
        self.assertEqual(dinputs.shape, dvalues.shape)
        
        # Ensure the values returned from backward are the same as dvalues (since the softmax
        # backward method doesn't do anything special, it's just passing through the gradient).
        np.testing.assert_array_equal(dinputs, dvalues)

    def test_forward_numerical_stability(self):
        softmax = Softmax()

        # Test with inputs having large values (for numerical stability)
        inputs = np.array([[1000, 1000, 1000],
                           [1000, 1000, 1000]])

        # Compute the softmax output
        output = softmax.forward(inputs)

        # Check if output is a valid probability distribution
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
        
        # Check if the sum of probabilities for each row is 1
        row_sums = np.sum(output, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_invalid_input(self):
        softmax = Softmax()

        # Test with invalid input (e.g., NaNs or Infs)
        inputs = np.array([[np.nan, 1.0, 2.0],
                           [1.0, np.inf, 3.0]])

        # Expect forward pass to raise an exception due to invalid values
        with self.assertRaises(ValueError):
            softmax.forward(inputs)


if __name__ == '__main__':
    unittest.main()
