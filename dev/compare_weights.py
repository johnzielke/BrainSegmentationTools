import h5py
import torch
import numpy as np
from brain_segmentation_tools.easyreg.model import EasyRegDeformableNet

def compare():
    h5_path = 'dev/freesurfer/mri_easyreg/easyreg_v10_230103.h5'
    
    # Correct H5 paths based on inspection
    tf_w_key = 'model_weights/vxm_dense/vxm_dense_unet_enc_conv_0_0/kernel:0'
    tf_b_key = 'model_weights/vxm_dense/vxm_dense_unet_enc_conv_0_0/bias:0'
    
    with h5py.File(h5_path, 'r') as f:
        tf_weight = f[tf_w_key][:]
        tf_bias = f[tf_b_key][:]

    model = EasyRegDeformableNet()
    model.load_from_tensorflow(h5_path)
    
    # In EasyRegDeformableNet, unet.encoder[0] is typically a Sequential containing ConvBlocks
    # Based on common structures, it might be unet.encoder[0][0].conv
    # Let's verify the actual path in the model
    layer = model.unet.encoder[0][0].conv
    pt_weight = layer.weight.detach().numpy()
    pt_bias = layer.bias.detach().numpy()

    # TF kernel shape: (f1, f2, f3, in_c, out_c)
    # PT kernel shape: (out_c, in_c, f1, f2, f3)
    tf_weight_transposed = tf_weight.transpose(4, 3, 0, 1, 2)

    print(f"TF Weight shape: {tf_weight.shape} -> Transposed: {tf_weight_transposed.shape}")
    print(f"PT Weight shape: {pt_weight.shape}")
    
    max_diff_w = np.max(np.abs(tf_weight_transposed - pt_weight))
    mean_diff_w = np.mean(np.abs(tf_weight_transposed - pt_weight))
    
    print(f"Weight - Max Abs Diff: {max_diff_w}, Mean Abs Diff: {mean_diff_w}")
    
    print(f"TF Bias shape: {tf_bias.shape}")
    print(f"PT Bias shape: {pt_bias.shape}")
    
    max_diff_b = np.max(np.abs(tf_bias - pt_bias))
    mean_diff_b = np.mean(np.abs(tf_bias - pt_bias))
    
    print(f"Bias - Max Abs Diff: {max_diff_b}, Mean Abs Diff: {mean_diff_b}")

if __name__ == "__main__":
    compare()
