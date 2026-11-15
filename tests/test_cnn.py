import os
import pytest
import numpy as np

def test_model_weights_exist():
    assert os.path.exists("model_weights.npz"), "Model weights file missing"
    weights = np.load("model_weights.npz")
    assert len(weights.files) > 0, "Model weights must contain arrays"

def test_sample_images_exist():
    assert os.path.exists("sample_images"), "sample_images directory missing"
    samples = os.listdir("sample_images")
    assert len(samples) > 0, "sample_images should not be empty"
