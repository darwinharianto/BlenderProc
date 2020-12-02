import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
import numpy as np
from src.camera.CameraInterface import CameraInterface
from src.utility.ItemCollection import ItemCollection


class CameraLoader(CameraInterface):
    """ Loads camera poses from the configuration and sets them as separate keypoints.
        Camera poses can be specified either directly inside the config or in an extra file.

        Example 1: Loads camera poses from file <args:0>, followed by the pose file format and setting the fov in radians.

        {
          "module": "camera.CameraLoader",
          "config": {
            "path": "<args:0>",
            "file_format": "location rotation/value",
            "default_cam_param": {
              "fov": 1
            }
          }
        }

        Example 2: More examples for parameters in "default_cam_param". Here cam_K is a camera matrix. Check
                   CameraInterface for more info on "default_cam_param".

        "default_cam_param": {
          "fov_is_half": true,
          "interocular_distance": 0.05,
          "stereo_convergence_mode": "PARALLEL",
          "convergence_distance": 0.00001,
          "cam_K": [650.018, 0, 637.962, 0, 650.018, 355.984, 0, 0 ,1],
          "resolution_x": 1280,
          "resolution_y": 720
        }

    **Configuration**:

    .. csv-table::
       :header: "Parameter", "Description"

       "cam_poses", "Optionally, a list of dicts, where each dict specifies one cam pose. See the next table for which "
                    "properties can be set. Type: list of dicts. Default: []."
       "path", "Optionally, a path to a file which specifies one camera position per line. The lines has to be "
               "formatted as specified in 'file_format'. Type: string. Default: ""."
       "file_format", "A string which specifies how each line of the given file is formatted. The string should contain "
                      "the keywords of the corresponding properties separated by a space. See next table for allowed "
                      "properties. Type: string. Default: ""."
       "default_cam_param", "A dictionary containing camera intrinsic parameters. Type: dict. Default: {}."
    """

    def __init__(self, config):
        CameraInterface.__init__(self, config)
        # A dict specifying the length of parameters that require more than one argument. If not specified, 1 is assumed.
        self.number_of_arguments_per_parameter = {
            "location": 3,
            "rotation/value": 3
        }
        self.cam_pose_collection = ItemCollection(self._add_cam_pose, self.config.get_raw_dict("default_cam_param", {}))

    def run(self):
        self.cam_pose_collection.add_items_from_dicts(self.config.get_list("cam_poses", []))
        self.cam_pose_collection.add_items_from_file(self.config.get_string("path", ""),
                                                     self.config.get_string("file_format", ""),
                                                     self.number_of_arguments_per_parameter)
        
        import json
        KRT = self.get_K_P_from_blender(bpy.context.scene.camera)
        with open('./camera.txt', 'w') as f:
            text = "" 
            for i in range(len(KRT["K"])):
                text += f"{KRT['K'][i][0]} {KRT['K'][i][1]} {KRT['K'][i][2]} \n"
            f.write(text)


    def _add_cam_pose(self, config):
        """ Adds new cam pose + intrinsics according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new cam pose.
        """

        # Collect camera object
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        # Set intrinsics and extrinsics from config
        self._set_cam_intrinsics(cam, config)
        self._set_cam_extrinsics(cam_ob, config)

        # Store new cam pose as next frame
        frame_id = bpy.context.scene.frame_end
        self._insert_key_frames(cam, cam_ob, frame_id)
        bpy.context.scene.frame_end = frame_id + 1


    # we could also define the camera matrix
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    def get_calibration_matrix_K_from_blender(self, camera):
        f_in_mm = camera.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = camera.sensor_width
        sensor_height_in_mm = camera.sensor_height
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if camera.sensor_fit == 'VERTICAL':
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
            s_v = resolution_y_in_px * scale / sensor_height_in_mm
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            s_u = resolution_x_in_px * scale / sensor_width_in_mm
            s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_u
        u_0 = resolution_x_in_px * scale / 2
        v_0 = resolution_y_in_px * scale / 2
        skew = 0  # only use rectangular pixels

        K = Matrix(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)))

        return K


    # Returns camera rotation and translation matrices from Blender.
    #
    # There are 3 coordinate systems involved:
    #    1. The World coordinates: "world"
    #       - right-handed
    #    2. The Blender camera coordinates: "bcam"
    #       - x is horizontal
    #       - y is up
    #       - right-handed: negative z look-at direction
    #    3. The desired computer vision camera coordinates: "cv"
    #       - x is horizontal
    #       - y is down (to align to the actual pixel coordinates
    #         used in digital images)
    #       - right-handed: positive z look-at direction
    def get_3x4_RT_matrix_from_blender(self, camera):
        # bcam stands for blender camera
        R_bcam2cv = Matrix(
            ((1, 0,  0),
            (0, -1, 0),
            (0, 0, -1)))

        # Use matrix_world instead to account for all constraints
        location, rotation = camera.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

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


    def get_3x4_P_matrix_from_blender(self, camera):
        K = self.get_calibration_matrix_K_from_blender(camera.data)
        RT = self.get_3x4_RT_matrix_from_blender(camera)
        return K*RT


    def get_K_P_from_blender(self, camera):
        K = self.get_calibration_matrix_K_from_blender(camera.data)
        RT = self.get_3x4_RT_matrix_from_blender(camera)
        return {"K": np.asarray(K, dtype=np.float32).tolist(), "RT": np.asarray(RT, dtype=np.float32).tolist()}