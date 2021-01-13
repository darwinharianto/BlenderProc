import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from mathutils import Matrix
from mathutils import Vector
import json
from tqdm import tqdm

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]

    return xy

def custom_to_coco(data_root, iterator):

    # K = np.loadtxt(os.path.join(data_root, '../../camera.txt'))
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))
    corner_3d = np.loadtxt(os.path.join(data_root, 'corner3d.txt'))

    model_meta = {
        'K': K,
        'fps_3d': fps_3d,
        'data_root': data_root,
        'corner_3d': corner_3d,
    }

    with open(f'./output_{iterator}/coco_data/coco_annotations.json', 'r') as f:
        coco_json = json.loads(f.read())

    new_coco_json = record_ann(model_meta, coco_json, iterator)
    anno_path = os.path.join(data_root, f'./output_{iterator}/coco_data/new_coco_annotations3.json')
    with open(anno_path, 'w') as f:
        json.dump(new_coco_json, f)



def get_3x4_RT_matrix_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    # location, rotation = matrix_world.decompose()[0:2]
    R_world2bcam = Matrix(rotation)

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],),
                R_world2cv[1][:] + (T_world2cv[1],),
                R_world2cv[2][:] + (T_world2cv[2],)))
    return RT

def record_ann(model_meta, coco_json, iterator):
    data_root = model_meta['data_root']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']
    corner_3d = model_meta['corner_3d']
    center_3d = np.mean(model_meta['corner_3d'], axis=0)

    # get object pose
    pose_path_obj = os.path.join(data_root, f"nihonbashi_positions_{iterator}")
    pose_obj = np.loadtxt(pose_path_obj)

    # get cam pose
    pose_path_cam = os.path.join(data_root, f"camera_positions_{iterator}")
    pose_cam = np.loadtxt(pose_path_cam)

    # iterate over frame for each pose


    for pose_iterator in tqdm(range(len(pose_cam))):
    
        pose_trans_obj = pose_obj[pose_iterator][:3]
        pose_rot_obj = R.from_euler('xyz', pose_obj[pose_iterator][3:])

        object_to_world_pose = np.append(pose_rot_obj.as_matrix(), pose_trans_obj.reshape(3,-1),axis=1)
        object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)


        RT = get_3x4_RT_matrix_from_blender(Vector(pose_cam[pose_iterator][:3]), R.from_euler('xyz', -pose_cam[pose_iterator][3:]).as_matrix())
        world_to_camera_pose = np.append(RT, [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]

        # calculate kpt_2d
        fps_2d = project(fps_3d, K, world_to_camera_pose)
        corner_2d = project(corner_3d, K, world_to_camera_pose)
        center_2d = project(center_3d, K, world_to_camera_pose)

        rgb_path = os.path.join(data_root, f'/output_{iterator}/coco_data/rgb_{str(pose_iterator).zfill(4)}.png')
        
        # search for category id This need fixes
        for category in coco_json["categories"]:
            if category["name"] == "1":
                category.update({
                    "keypoints": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "skeleton": []
                })

        # search for image id in json annot
        for image in coco_json["images"]:
            if image["file_name"] == rgb_path.split('/')[-1]:
                image_id = image["id"]
                break
        


        for annot in coco_json["annotations"]:
            # add kpt on annotations
            if annot["image_id"] == image_id:
                kpt_2d_flat = fps_2d.flatten().tolist()
                kpt_3d_flat = fps_3d.flatten().tolist()
                i = 2
                while i <= len(kpt_2d_flat):
                    kpt_2d_flat.insert(i, 2)
                    i += 3
                i=3
                while i <= len(kpt_3d_flat):
                    kpt_3d_flat.insert(i, 2)
                    i += 4
                annot.update({"keypoints": kpt_2d_flat, 
                              "num_keypoints": len(fps_2d),
                              "keypoints_3d": kpt_3d_flat,
                              "box_2d": corner_2d.flatten().tolist(),
                              "box_3d": corner_3d.flatten().tolist(),
                              "center_2d": center_2d.flatten().tolist(),
                              "center_3d": center_3d.flatten().tolist(),
                            })

            if annot["area"] < 100:

                print("BAD AREA, COULD CAUSE RPN ERROR")
        
        if True:
            import cv2
            image = cv2.imread(os.path.join(data_root, f"output_{iterator}/coco_data/rgb_{str(pose_iterator).zfill(4)}.png"))
            for item in fps_2d:
                image = cv2.circle(image, (int(item[0]), int(item[1])), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.imwrite('asd.png', image)
            # cv2.waitKey(0)
        
    return coco_json



if __name__ == "__main__":
    
    path = os.getcwd()

    for iterator in range(1,6):
        custom_to_coco(path, iterator)
