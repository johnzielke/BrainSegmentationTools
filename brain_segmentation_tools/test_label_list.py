from utils import get_list_labels

def test_n_unique_labels():

    combinations = [
        ["segmentation", "v2.0", 19],
        ["segmentation", "v1.0", 18],

    ]
    for model, version, n_neutral in combinations:
        label_list, n_neutral_cal = get_list_labels(model,version,FS_sort=True)
        assert n_neutral_cal == n_neutral

test_n_unique_labels()