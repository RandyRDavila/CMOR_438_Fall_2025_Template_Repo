import unittest
import numpy as np
import sys
import os

# 1. GET THE PATH TO THE 'src' FOLDER
# This goes up two levels from 'tests/unit' to the root, then points to 'src'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))

# 2. ADD IT TO PYTHON'S PATH
if src_path not in sys.path:
    sys.path.append(src_path)

# 3. NOW IMPORT YOUR MODULE (Must happen AFTER step 2)
from rice_ml.supervised_learning.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    
    def setUp(self):
        """
        Runs before every test. We create some simple synthetic data here.
        Equation: y = 2*x1 + 3*x2 + 4 + noise
        """
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 2
        
        # Random feature data
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # True coefficients: [2, 3] and Intercept: 4
        true_coefs = np.array([2.0, 3.0])
        true_intercept = 4.0
        
        # y = X @ coefs + intercept + small_noise
        noise = np.random.randn(self.n_samples) * 0.1 
        self.y = self.X @ true_coefs + true_intercept + noise

    def test_initialization(self):
        """Test that the model initializes with correct default values."""
        model = LinearRegression(fit_intercept=True)
        self.assertTrue(model.fit_intercept)
        self.assertIsNone(model.coef_)

    def test_simple_fit_exact(self):
        """
        Test fitting on a perfect line (no noise) to ensure math is exact.
        y = 5x + 10
        """
        X = np.array([[1], [2], [3], [4]])
        y = np.array([15, 20, 25, 30]) # Slope=5, Intercept=10
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Check coefficients (Slope)
        self.assertAlmostEqual(model.coef_[0], 5.0, places=5)
        # Check intercept
        self.assertAlmostEqual(model.intercept_, 10.0, places=5)
        
        # Check R2 is 1.0 for perfect fit
        self.assertAlmostEqual(model.R2(), 1.0, places=5)

    def test_multivariate_fit(self):
        """Test fitting on the multivariate 2D data created in setUp."""
        model = LinearRegression(fit_intercept=True)
        model.fit(self.X, self.y)
        
        # Check shapes
        self.assertEqual(model.coef_.shape, (2,))
        
        # Check that learned coefficients are close to true values (2 and 3)
        # Note: We use a larger delta because of the random noise added
        self.assertAlmostEqual(model.coef_[0], 2.0, delta=0.2)
        self.assertAlmostEqual(model.coef_[1], 3.0, delta=0.2)
        self.assertAlmostEqual(model.intercept_, 4.0, delta=0.2)

    def test_predict_dimensions(self):
        """Test that predict returns the correct shape (1D array)."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        preds = model.predict(self.X)
        
        # Your code specifies predict returns a 1D array using .ravel()
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(preds.shape[0], self.n_samples)

    def test_no_intercept(self):
        """Test the fit_intercept=False functionality."""
        # Data through origin: y = 3x
        X = np.array([[1], [2], [3]])
        y = np.array([3, 6, 9])
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        self.assertEqual(model.intercept_, 0.0)
        self.assertAlmostEqual(model.coef_[0], 3.0, places=5)

    def test_metrics(self):
        """Test internal calculation of MSE, SSE, SST."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        mse = model.MSE()
        sse = model.SSE()
        sst = model.SST()
        r2 = model.R2()
        
        # Basic logic checks
        self.assertGreater(mse, 0)
        self.assertGreater(sse, 0)
        self.assertGreater(sst, sse) # Total variance should be > residual variance (usually)
        self.assertTrue(0 <= r2 <= 1)

    def test_errors_when_not_fitted(self):
        """Ensure methods raise ValueError if called before .fit()."""
        model = LinearRegression()
        
        with self.assertRaises(ValueError):
            model.predict(self.X)
            
        with self.assertRaises(ValueError):
            model.summary()

    def test_summary_runs(self):
        """Simply check that summary() prints without crashing."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        try:
            # We don't check the text output, just that it executes
            model.summary() 
        except Exception as e:
            self.fail(f"model.summary() raised exception: {e}")

if __name__ == '__main__':
    unittest.main()