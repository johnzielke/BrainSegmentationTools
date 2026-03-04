Registration using synthseg

# Run synthseg on all your scans
Example if you have to files with each line being a path to a nifti file and the output path of the segmentation

mri_synthseg --parc --i synthseg/input_paths.txt --o synthseg/output_paths.txt


# Prepare your atlas
Find a nifti file of the atlas you want to register to and run synthseg on it as well

mri_synthseg --parc --i myatlas.nii.gz --o myatlas_synthseg.nii.gz

# Run the registration based on these values