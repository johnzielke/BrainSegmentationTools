import pytest

from brain_segmentation_tools import utils


@pytest.mark.parametrize(
    ("model_name", "version", "expected_neutral_count"),
    [
        ("segmentation", "v2.0", 5),
        ("segmentation", "v1.0", 4),
    ],
)
def test_get_list_labels_sorted_returns_expected_neutral_count(
    model_name: str, version: str, expected_neutral_count: int
) -> None:
    _label_list, neutral_count = utils.get_list_labels_sorted(
        model_name,
        version,
        FS_sort=True,
    )
    assert neutral_count == expected_neutral_count
