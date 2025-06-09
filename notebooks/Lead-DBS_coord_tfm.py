import numpy as np
import pandas as pd
import scipy.io
import os
import SimpleITK as sitk

def apply_affine_transform(mat_path, fcsv_path):
    """
    Applies a transformation from a .mat file to coordinates in a .fcsv file
    
    Parameters:
    - fcsv_path: str, path to the .fcsv file with coordinates
    - mat_path: str, path to the .mat file with the transformation matrix
    - output_path: str, path to save the transformed coordinates file
    
    Returns:
    - None, saves the transformed coordinates to the specified output path
    """
    
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_path)
    
    # Extract the transformation matrix
    transformation_matrix = mat_contents['tmat']
    
    # Read the .fcsv file with correct delimiter and header handling
    fcsv_data = pd.read_csv(fcsv_path, delimiter=',', header=2)  # Skipping first two lines which are presets (Assuming all coordinates are in RAS)
    
    # Extract the coordinates
    coordinates = fcsv_data[['x', 'y', 'z']].values
    
    # Convert the 3D coordinates to homogeneous coordinates
    homogeneous_coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    
    # Apply the 4x4 transformation matrix to the homogeneous coordinates
    transformed_homogeneous_coordinates = (np.linalg.inv(transformation_matrix) @ homogeneous_coordinates.T).T
    
    # Convert back to 3D coordinates by discarding the homogeneous component
    transformed_coordinates = transformed_homogeneous_coordinates[:, :3]
    
    return transformed_coordinates

def apply_warp_deformation(transform_path, fiducial_file):
    """
    Transforms fiducial points from a fiducial file using a transformation matrix.

    Parameters:
    - transform_path: str, path to the transformation matrix file
    - fiducial_file: str, path to the fiducial file

    Returns:
    - transformed_fiducial_points: list of transformed fiducial points
    - fiducial_properties: list of properties corresponding to each fiducial point
    """
    
    # Reads the transform and casts the output to a compatible format
    transform_image = sitk.ReadImage(transform_path)
    transform_image = sitk.Cast(transform_image, sitk.sitkVectorFloat64)

    # Load it as a transform
    transform = sitk.Transform(transform_image)

    # Loop through the file and extract the fiducial points
    fiducial_points = []
    with open(fiducial_file, 'r') as file:
        for line in file.readlines():
            # Skip comment lines starting with '#'
            if not line.startswith('#'):
                # Extract the properties and coordinates from each line
                fields = line.strip().split(',')
                x, y, z = float(fields[1]), float(fields[2]), float(fields[3])
                fiducial_points.append([x, y, z])

    fiducial_points = np.array(fiducial_points) * np.array([-1, -1, 1])

    # Apply the transform to the fiducial points
    transformed_fiducial_points = []
    for point in fiducial_points:
        transformed_point = transform.TransformPoint(point.tolist())
        transformed_fiducial_points.append(transformed_point)

    transformed_fiducial_points = np.array(transformed_fiducial_points) * np.array([-1, -1, 1])

    return transformed_fiducial_points


def coords_to_fcsv(coords_array, fcsv_output, target):

    if target == 'CM':
        fcsv = [
            '# Markups fiducial file version = 4.10',
            '# CoordinateSystem = RAS',
            '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID',
            '1,afid1_x,afid1_y,afid1_z,0,0,0,1,1,1,1,1,R_CM,vtkMRMLScalarVolumeNode1',
            '2,afid2_x,afid2_y,afid2_z,0,0,0,1,1,1,1,2,L_CM,vtkMRMLScalarVolumeNode1'
        ]
    elif target == 'ANT':
        fcsv = [
            '# Markups fiducial file version = 4.10',
            '# CoordinateSystem = RAS',
            '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID',
            '1,afid1_x,afid1_y,afid1_z,0,0,0,1,1,1,1,1,R_ANT,vtkMRMLScalarVolumeNode1',
            '2,afid2_x,afid2_y,afid2_z,0,0,0,1,1,1,1,2,L_ANT,vtkMRMLScalarVolumeNode1'
        ] 

    # Loop over fiducials
    for fid in range(1, coords_array.shape[0]+1):
        # Update fcsv, skipping header
        line_idx = fid + 2
        centroid_idx = fid - 1
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_x", str(coords_array[centroid_idx][0])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_y", str(coords_array[centroid_idx][1])
        )
        fcsv[line_idx] = fcsv[line_idx].replace(
            f"afid{fid}_z", str(coords_array[centroid_idx][2])
        )

    # Ensure the directory exists
    create_folder(os.path.dirname(fcsv_output))
    # Write output fcsv
    with open(fcsv_output, "w") as f:
        f.write("\n".join(line for line in fcsv))
        
        
def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")


ANALYSES = ['leaddbs'] #multiple renamed leaddbs derivative folders; each is using different images for registration 

BASE_DIR = '/home/UWO/dbansal7/graham/scratch/DBSepilepsy/derivatives'

for ANALYSIS in ANALYSES:
    
    SUBJECT_IDS = [subject for subject in os.listdir(f'{BASE_DIR}/{ANALYSIS}') if "sub-" in subject]

    print(len(SUBJECT_IDS),SUBJECT_IDS)

    for subject in SUBJECT_IDS:
        if subject in ['P342', 'P307']:
            break

        if subject in ['P329','P280','P248','P238','P333','P282','P267']:
            target = 'CM'
        else:
            target = 'ANT'

        print(subject)

        transform_affine_path = f'{BASE_DIR}/{ANALYSIS}/{subject}/coregistration/transformations/{subject}_desc-precoreg_ax_T1w.mat' #specify moving to fixed bc function inverts the matrix
        transform_warp_path = f'{BASE_DIR}/{ANALYSIS}/{subject}/normalization/transformations/{subject}_from-anchorNative_to-MNI152NLin2009bAsym_desc-ants.nii.gz' #because we need to take coordinate from native space to MNI we read the inverse matrix
        fiducial_file = f'../cleaned_AT/{target}/sub-{subject}_desc-electrodefiducials.fcsv'
        
        transformed_coordinates = apply_warp_deformation(transform_warp_path, fiducial_file)

        if os.path.exists(transform_affine_path):
            print("found anchor native transformation")
            # Load the .mat file
            mat_contents = scipy.io.loadmat(transform_affine_path)
            # Extract the transformation matrix
            transformation_matrix = mat_contents['tmat']
            # Convert the 3D coordinates to homogeneous coordinates
            homogeneous_coordinates = np.hstack((transformed_coordinates, np.ones((transformed_coordinates.shape[0], 1))))
            # Apply the 4x4 transformation matrix to the homogeneous coordinates
            transformed_homogeneous_coordinates = (transformation_matrix @ homogeneous_coordinates.T).T
            # Convert back to 3D coordinates by discarding the homogeneous component
            transformed_coordinates = transformed_homogeneous_coordinates[:, :3]

        deform_fcsv = f'../transformed/{target}/sub-{subject}_desc-electrodefiducials_space-native.fcsv'
        coords_to_fcsv(transformed_coordinates,deform_fcsv, target)