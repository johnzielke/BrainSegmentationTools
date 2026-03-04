import numpy as np
from monai.networks.layers import AffineTransform
from monai import transforms, metrics
from monai.utils.type_conversion import convert_to_numpy, convert_to_dst_type, convert_to_tensor
import monai
from dataclasses import dataclass
from monai.utils import GridSampleMode
from numba import jit
import torch
from monai.utils.enums import StrEnum

SYNTHSEG_LABELS = np.array([2,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,43,44,46,47,49,50,51,52,53,54,58,60,
                                    1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,
                                    2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035])
N_SYNTHSEG_LABELS = len(SYNTHSEG_LABELS)



REMAP_VECTOR = np.zeros(np.max(SYNTHSEG_LABELS)+1, dtype=np.int32)
for i, label in enumerate(SYNTHSEG_LABELS):
    REMAP_VECTOR[label] = i + 1




@jit(nopython=True, nogil=True)
def _calculate_COG_numba(segmentation) -> tuple[np.ndarray, np.ndarray]:
    n_labels = N_SYNTHSEG_LABELS + 1
    sum_coordinates = np.zeros((n_labels, 3),dtype=np.uint64)
    segment_volumes = np.zeros((n_labels,1),dtype=np.uint32)
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            for k in range(segmentation.shape[2]):
                label = segmentation[i,j,k]
                sum_coordinates[label, 0] += i
                sum_coordinates[label, 1] += j
                sum_coordinates[label, 2] += k
                segment_volumes[label,0] += 1
    cog = np.ones((4, n_labels),dtype=np.float32)
    cog[:3, :] = (sum_coordinates / segment_volumes.astype(np.float64)).T.astype(np.float32)
    return cog[:,1:], segment_volumes[1:,0]



def _calculate_COG_numpy(segmentation, use_mean=False):
    cog = np.zeros([4, N_SYNTHSEG_LABELS])
    segment_volumes = np.zeros(N_SYNTHSEG_LABELS,dtype=np.int32)
    reduction = np.mean if use_mean else np.median
    for l, label in enumerate(SYNTHSEG_LABELS):
        aux = np.where(segmentation == label)
        
        cog[0, l] = reduction(aux[0])
        cog[1, l] = reduction(aux[1])
        cog[2, l] = reduction(aux[2])
        cog[3, l] = 1
        segment_volumes[l] = len(aux[0])
    return cog, segment_volumes

def calculate_COG(segmentation: np.ndarray, affine: np.ndarray | None=None, use_mean: bool =False):
    """
    Calculate the center of gravity of the segmentation
    Args:
        segmentation: segmentation image. A labelmap with the same shape as the image
        affine: affine transformation matrix of the image
        use_mean: use mean or median to calculate the center of gravity
    Returns:
        cog: np.ndarray center of gravity coordinates (N_SYNTHSEG_LABELS, 3)
        segment_volumes: np.ndarray volumes of the segments (N_SYNTHSEG_LABELS,)
    """
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError(f"Segmentation must be an integer type, found {segmentation.dtype}")
    if affine is None:
        affine = np.eye(4)
    if use_mean:
        remap_segmentation = REMAP_VECTOR[segmentation]
        cog, segment_volumes = _calculate_COG_numba(remap_segmentation)
    else:
        cog, segment_volumes = _calculate_COG_numpy(segmentation, use_mean)
    cog = np.matmul(affine, cog)[:-1, :]
    return cog.T, segment_volumes


def find_affine_matrix(ref, mov):
    """ Find the affine matrix that maps mov to ref
    Args:
        ref: reference image points (N, 3)
        mov: moving image points (N, 3)


    """
    # Convert lists to numpy arrays  
    P = ref
    Q = mov
  
    # Add a column of ones to P to create the augmented matrix  
    n = P.shape[0]  
    P_aug = np.hstack([P, np.ones((n, 1))])  
  
    # Solve for the affine transformation matrix using least squares  
    A, res, rank, s = np.linalg.lstsq(P_aug, Q, rcond=None)  
    # print(A.shape)
    # print(A)
    # Adding the row [0, 0, 0, 1] to make it a 4x4 affine matrix  
    affine_matrix = np.concatenate([A.T, [[0, 0, 0, 1]]],axis=0)  
  
    return affine_matrix 




def find_rigid_matrix(ref, mov):
    """ Find the rigid transformation matrix that maps mov to ref
    Args:
        ref: reference image points (N, 3)
        mov: moving image points (N, 3)
    Returns:
        rigid_matrix: 4x4 rigid transformation matrix
    """
    # Convert lists to numpy arrays  
    P = ref
    Q = mov

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    t = centroid_Q - np.dot(R, centroid_P)

    # Construct the 4x4 rigid transformation matrix
    rigid_matrix = np.eye(4)
    rigid_matrix[:3, :3] = R
    rigid_matrix[:3, 3] = t

    return rigid_matrix 




@dataclass
class RegistrationAtlas:
    """
    A class used to represent a Registration Atlas.
    Attributes
    ----------
    cogs :
        Center of gravity coordinates. (N_SYNTHSEG_LABELS,3)
    affine :
        Affine transformation matrix of the atlas. (4,4)
    shape :
        Shape of the atlas. (3,)
    """

    cogs: np.ndarray
    affine: np.ndarray
    shape: tuple[int, int, int]


class RegistrationMethod(StrEnum):

    RIGID = "rigid"
    AFFINE = "affine"

class RegisterToAtlas(transforms.transform.Transform):
    MIN_SEGMENT_VOLUME = 50

    def __init__(self, *,
                  cog_key,
                  brain_segment_volume_key,
                  image_keys = tuple(),
                  segmentation_keys = tuple(),
                  registration_atlas: RegistrationAtlas,
                  affine_output_key = "registration_affine",
                  registration_method: RegistrationMethod = RegistrationMethod.AFFINE,
                  **kwargs):
        super().__init__(**kwargs)
        # self.image_keys = image_keys
        # self.segmentation_keys = segmentation_keys
        # if len(self.image_keys) == 0 and len(self.segmentation_keys) == 0:
        #     raise ValueError("Either image_keys or segmentation_keys must be provided")
        self.cog_key = cog_key
        self.brain_segment_volume_key = brain_segment_volume_key
        self.registration_atlas = registration_atlas
        self.affine_output_key = affine_output_key
        self.registration_method = registration_method
        if image_keys or segmentation_keys:
            self.resampling_transform = ApplyRegistrationAffineTransform(
                image_keys=image_keys,
                segmentation_keys=segmentation_keys,
                registration_atlas=registration_atlas,
                affine_key=affine_output_key
            )
        else:
            self.resampling_transform = None
    
    def find_registraion_matrix(self, ref, mov):
        if self.registration_method == RegistrationMethod.RIGID:
            return find_rigid_matrix(ref, mov)
        elif self.registration_method == RegistrationMethod.AFFINE:
            return find_affine_matrix(ref, mov)
        else:
            raise ValueError(f"Unknown registration method: {self.registration_method}")

    def __call__(self, data):
        result = dict(data)
        segment_volumes = np.array(data[self.brain_segment_volume_key])
        cogs = np.array(data[self.cog_key])
        valid_cogs = (segment_volumes > self.MIN_SEGMENT_VOLUME)
        transform_matrix = self.find_registraion_matrix(
            np.array(self.registration_atlas.cogs)[valid_cogs,:],
            cogs[valid_cogs,:]
        )
        if self.affine_output_key is not None:
            result[self.affine_output_key] = transform_matrix
        
        if self.resampling_transform is not None:
            result = self.resampling_transform(result, transform_matrix)

        return result

class ApplyRegistrationAffineTransform(transforms.transform.Transform):

    def __init__(self, *,
                  image_keys = tuple(),
                  segmentation_keys = tuple(),
                  registration_atlas: RegistrationAtlas| None = None,
                  registration_shape: tuple[int,int,int]| None = None,
                  registration_affine: np.ndarray| None = None,
                  affine_key = "registration_affine"
        ):
        self.image_keys = image_keys
        self.segmentation_keys = segmentation_keys
        if len(self.image_keys) == 0 and len(self.segmentation_keys) == 0:
            raise ValueError("Either image_keys or segmentation_keys must be provided")
        if registration_atlas is None:
            if registration_shape is None or registration_affine is None:
                raise ValueError("Either registration_atlas or both registration_shape and registration_affine must be provided")
            self.registration_shape = registration_shape
            self.registration_affine = registration_affine
        else:
            if registration_shape is not None or registration_affine is not None:
                raise ValueError("If registration_atlas is provided, registration_shape and registration_affine must be None")
            self.registration_shape = registration_atlas.shape
            self.registration_affine = registration_atlas.affine
        self.affine_key = affine_key
    
    def __call__(self, data, transform_matrix=None):
        result = dict(data)
        if transform_matrix is None:
            transform_matrix = data[self.affine_key]
        def transform_tensor(affine_transform, tensor):
            tensor = convert_to_tensor(tensor, track_meta=True)
            affine = np.matmul(np.linalg.inv(convert_to_numpy(tensor.affine)),np.matmul(transform_matrix, self.registration_affine))
            converted_affine = convert_to_dst_type(affine, tensor)[0]
            res = affine_transform(tensor[None], converted_affine)[0]
            return monai.data.MetaTensor(res,meta={**result[key].meta, "affine":self.registration_affine})

        if len(self.image_keys) > 0:
            aff_transform_image = AffineTransform(self.registration_shape,normalized=False)
            for key in self.image_keys:
                result[key] = transform_tensor(aff_transform_image, data[key])

        if len(self.segmentation_keys) > 0:
            aff_transform_segmentation = AffineTransform(self.registration_shape,normalized=False,mode=GridSampleMode.NEAREST)
            for key in self.segmentation_keys:
                result[key] = transform_tensor(aff_transform_segmentation, data[key])

        return result


class AtlasDiceCorrelation(transforms.transform.MapTransform):
    def __init__(self, atlas_segmentation, keys,
                  output_key_segmentation="atlas_dice",
                #   output_key_brain_mask="atlas_brain_mask_dice",
                    do_remapping=True):
        super().__init__(keys=keys)
        atlas_segmentation = torch.as_tensor(atlas_segmentation).to(dtype=torch.int32)
        self.classes = torch.unique(atlas_segmentation)
        self.n_classes = self.classes.numel()
        self.atlas_segmentation = atlas_segmentation[None,None]
        if do_remapping:
            self.atlas_segmentation = self.remap_segmentation(self.atlas_segmentation)
        self.output_key_segmentation = output_key_segmentation
        # self.output_key_brain_mask = output_key_brain_mask
        self.do_remapping = do_remapping
    def remap_segmentation(self, segmentation):
        result = torch.zeros_like(segmentation, dtype=torch.int32)
        for i in range(1,self.n_classes):
            result[segmentation == self.classes[i]] = i
        return result
        
    def __call__(self, data):
        d = dict(data)
        segmentation_dices = []
        # brain_mask_dices = []
        for key in self.key_iterator(data):
            segment_data = d[key].to(dtype=torch.int32)[None]
            if self.do_remapping:
                segment_data = self.remap_segmentation(segment_data)
            atlas_segment = self.atlas_segmentation.to(device=segment_data.device)
            segmentation_dices.append(metrics.compute_dice(segment_data, atlas_segment, include_background=False,num_classes=self.n_classes))
            # brain_mask_dices.append(metrics.compute_dice(segment_data > 0, atlas_segment > 0, include_background=False,num_classes=2))
        d[self.output_key_segmentation] = torch.stack(segmentation_dices)
        # d[self.output_key_brain_mask] = torch.stack(brain_mask_dices)
        return d
            
def remap_segmentation(segmentation):
    return monai.utils.convert_to_dst_type(REMAP_VECTOR, segmentation)[0][segmentation.to(dtype=torch.int)]
