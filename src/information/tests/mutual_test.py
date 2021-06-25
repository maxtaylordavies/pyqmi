import unittest
import numpy as np

from src.information.mutual import (
    quadratic,
    project_stimulus,
    project_stimulus_inverse,
    quadratic_first_grad,
)


class TestQMI(unittest.TestCase):
    def test_qmi(self):
        raw_stimulus = np.transpose(np.loadtxt("sample_stimulus.txt"))  # N x F
        response = np.loadtxt("sample_response.txt").reshape((1, 5987))
        rfv = np.loadtxt("sample_rfv.txt")
        sigma = 2.0119
        expected_qmi = 5.7201e-08

        projected_stimulus = project_stimulus(raw_stimulus, rfv, 5)
        got_qmi = quadratic(response, projected_stimulus, sigma)

        percentage_error = 100 * np.abs(got_qmi - expected_qmi) / expected_qmi
        self.assertTrue(percentage_error < 1)

    def test_inverse_projection(self):
        raw_stimulus = np.transpose(np.loadtxt("sample_stimulus.txt"))
        ps = np.loadtxt("sample_ps.txt")
        H = 5  # frame history

        rfv_estimate = project_stimulus_inverse(raw_stimulus, ps, H)
        sample_rfv_estimate = np.loadtxt("sample_estimate_rfv.txt")

        errors = np.abs(rfv_estimate - sample_rfv_estimate)
        percentage_error = np.sum(errors) / np.sum(np.abs(rfv_estimate))
        self.assertTrue(percentage_error < 1)

    def test_gradient(self):
        raw_stimulus = np.transpose(np.loadtxt("sample_stimulus.txt"))  # N x F
        response = np.loadtxt("sample_response.txt").reshape((1, 5987))
        H = 5  # frame history
        sigma = 2.0851

        rfv = np.loadtxt("sample_rfv.txt")
        projected_stimulus = project_stimulus(raw_stimulus, rfv, H)

        gradient = quadratic_first_grad(
            raw_stimulus, response, projected_stimulus, sigma, H
        )
        sample_gradient = np.loadtxt("sample_drfv_norm.txt")

        errors = np.abs(gradient - sample_gradient)
        percentage_error = np.sum(errors) / np.sum(np.abs(gradient))
        self.assertTrue(percentage_error < 1)


if __name__ == "__main__":
    unittest.main()

