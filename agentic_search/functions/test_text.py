import os
import pytest
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from functions.text import normalize_and_lemmatize_text


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("US Politics/International Relations", "us politics international relation"),
        ("Technology/Finance", "technology finance"),
        ("Financial Markets", "financial market"),
        ("Business/Economy", "business economy"),
        ("Technology/Economy", "technology economy"),
        ("Business/Finance/Economy", "business finance economy"),
        (
            "Technology/Computing - Artificial Intelligence",
            "technology computing artificial intelligence",
        ),
        ("Business/Finance - Insurance", "business finance insurance"),
        ("Automotive Industry", "automotive industry"),
    ],
)
def test_normalize_and_lemmatize_text(input_text, expected_output):
    assert normalize_and_lemmatize_text(input_text) == expected_output
