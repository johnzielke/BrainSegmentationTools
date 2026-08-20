import h5py

def print_structure(name, obj):
    print(name)

h5_path = 'dev/freesurfer/mri_easyreg/easyreg_v10_230103.h5'
with h5py.File(h5_path, 'r') as f:
    f.visititems(print_structure)
