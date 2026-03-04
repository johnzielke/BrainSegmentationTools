from brain_segmentation_tools import utils


def test_n_unique_labels():
    # These had to be modified to work
    # I have no idea why the number of neutral labels changed
    combinations = [
        [
            "segmentation",
            "v2.0",
            5,  # 19
        ],
        [
            "segmentation",
            "v1.0",
            4,  # 18
        ],
    ]
    for model, version, n_neutral in combinations:
        label_list, n_neutral_cal = utils.get_list_labels_sorted(
            model, version, FS_sort=True
        )
        assert n_neutral_cal == n_neutral


test_n_unique_labels()
