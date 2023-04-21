import unittest

import data_utils
import numpy as np


class TestSegmentEEG(unittest.TestCase):
    def test_segment_eeg_trivial(self):
        # Arrange
        eeg = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ])

        sample_rate = 1
        segment_len_s = 1
        overlap_s = 0

        expected_output = [
            np.array([[0, 1]]),
            np.array([[2, 3]]),
            np.array([[4, 5]]),
            np.array([[6, 7]]),
            np.array([[8, 9]]),
        ]

        # Act
        output = data_utils.segment_eeg(eeg, sample_rate, segment_len_s, overlap_s)

        # Assert
        self.assertEqual(len(expected_output), len(output))
        for i in range(len(expected_output)):
            np.testing.assert_array_equal(expected_output[i], output[i])

    def test_segment_eeg_simple(self):
        # Arrange
        eeg = np.arange(15).reshape((5, 3))  # 5 samples, 3 channels
        sample_rate = 1
        segment_len_s = 2
        overlap_s = 1

        expected_output = [
            np.array([[0, 1, 2], [3, 4, 5]]),
            np.array([[3, 4, 5], [6, 7, 8]]),
            np.array([[6, 7, 8], [9, 10, 11]]),
            np.array([[9, 10, 11], [12, 13, 14]]),
        ]

        # Act
        output = data_utils.segment_eeg(eeg, sample_rate, segment_len_s, overlap_s)

        # Assert
        self.assertEqual(len(expected_output), len(output), f"Lens: {list(map(len, output))}")
        for i in range(len(expected_output)):
            np.testing.assert_array_equal(output[i], expected_output[i])

    def test_segment_eeg_realistic(self):
        # Arrange
        # Create an EEG signal with 1100 samples and 2 channels
        eeg = np.random.randn(1100, 2)

        # Segment the EEG signal into 250 ms segments with 100 ms overlap
        sample_rate = 1000
        segment_len_s = 0.25
        overlap_s = 0.1

        # Expected: 0-250, 150-400, 300-550, 450-700, 600-850, 750-1000, and drop 1000-1100
        expected_num_segments = 6
        expected_segment_len = 250

        # Act
        segments = data_utils.segment_eeg(eeg, sample_rate, segment_len_s, overlap_s)

        # Assert
        self.assertEqual(len(segments), expected_num_segments, f"Lens: {list(map(len, segments))}")
        for segment in segments:
            self.assertEqual(expected_segment_len, len(segment))
            self.assertEqual(2, segment.shape[1])

