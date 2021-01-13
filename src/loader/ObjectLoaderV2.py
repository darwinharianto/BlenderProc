import bpy

from src.main.Module import Module
from src.loader.LoaderInterface import LoaderInterface
from src.utility.BlenderUtility import check_intersection, check_bb_intersection, get_all_mesh_objects
from src.utility.ItemCollection import ItemCollection
from src.utility.Utility import Utility
from src.utility.MathUtility import MathUtility
from mathutils import Matrix, Vector, Euler
import numpy as np


class ObjectLoaderV2(LoaderInterface):
    """ Loads object poses from the configuration and sets them as separate keypoints.

        Example 1: Loads object poses from file <args:0>, followed by the pose file format and setting the fov in radians.

        {
          "module": "object.ObjectLoader",
          "config": {
            "path": "<args:0>",
            "file_format": "location rotation/value",
          }
        }


    **Configuration**:

    .. csv-table::
       :header: "Parameter", "Description"

       "path", "Optionally, a path to a file which specifies one camera position per line. The lines has to be "
               "formatted as specified in 'file_format'. Type: string. Default: ""."
       "file_format", "A string which specifies how each line of the given file is formatted. The string should contain "
                      "the keywords of the corresponding properties separated by a space. See next table for allowed "
                      "properties. Type: string. Default: ""."
    """

    def __init__(self, config):
        Module.__init__(self, config)
        # A dict specifying the length of parameters that require more than one argument. If not specified, 1 is assumed.
        self.number_of_arguments_per_parameter = {
            "location": 3,
            "rotation/value": 3
        }
        self.obj_pose_collection = ItemCollection(self._add_obj_pose, self.config.get_raw_dict("default_obj_param", {}))
        self.obj_name = self.config.get_string("obj_name", None)
        self.source_frame = self.config.get_list("source_frame", ["X", "Y", "Z"])
        self.child_name = self.config.get_list("child", [""])
        

    def run(self):
        self.obj_pose_collection.add_items_from_dicts(self.config.get_list("obj_poses", []))
        self.obj_pose_collection.add_items_from_file(self.config.get_string("path", ""),
                                                     self.config.get_string("file_format", ""),
                                                     self.number_of_arguments_per_parameter)
        
        self._set_properties([bpy.context.scene.objects[self.obj_name]])

        for child_name in self.child_name:
            if child_name != "":
                bpy.context.scene.objects[child_name].parent = bpy.context.scene.objects[self.obj_name]
        

    def _add_obj_pose(self, config):
        """ Adds new obj pose according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new obj pose.
        """

        # Collect camera object
        if self.obj_name is not None:

            obj_ob = bpy.context.scene.objects[self.obj_name]
            obj = obj_ob.data

            # Store new cam pose as next frame
            start_frame = bpy.context.scene.frame_start
            end_frame = bpy.context.scene.frame_end
                        
            self._set_obj_loc_rot(obj_ob, config)

            #TODO this will take too long, need to fix this
            if obj_ob.type in ['MESH','ARMATURE'] and obj_ob.animation_data:
                try:
                    filled_frame = []
                    for fc in obj_ob.animation_data.action.fcurves :
                        if fc.data_path.endswith(('location','rotation_euler','rotation_quaternion','scale')):
                            for key in fc.keyframe_points :
                                filled_frame.append(key.co[0])
                    
                    print(f"inserting new keyframe for obj at {int(max(filled_frame)+1)}")
                    self._insert_key_frames(obj, obj_ob, int(max(filled_frame))+1)

                except AttributeError:
                    self._insert_key_frames(obj, obj_ob, start_frame)

    def _insert_key_frames(self, obj, obj_ob, frame_id):
        """ Insert key frames for all relevant camera attributes.

        :param obj: The obj which contains only obj specific attributes.
        :param obj_ob: The object linked to the obj which determines general properties like location/orientation
        :param frame_id: The frame number where key frames should be inserted.
        """

        obj_ob.keyframe_insert(data_path='location', frame=frame_id)
        obj_ob.keyframe_insert(data_path='rotation_euler', frame=frame_id)

    def _set_obj_loc_rot(self, obj_ob, config):
        """ Sets obj extrinsics according to the config.

        :param obj_ob: The object linked to the obj which determines general properties like location/orientation
        :param config: A configuration object with cam extrinsics.
        """
        obj2world_matrix = self._obj2world_matrix_from_obj_loc_rot(config)
        obj_ob.matrix_world = obj2world_matrix

    
    def _obj2world_matrix_from_obj_loc_rot(self, config):
        """ Determines camera extrinsics by using the given config and returns them in form of a cam to world frame transformation matrix.

        :param config: The configuration object.
        :return: The cam to world transformation matrix.
        """
        if not config.has_param("obj2world_matrix"):
            position = MathUtility.transform_point_to_blender_coord_frame(config.get_vector3d("location", [0, 0, 0]), self.source_frame)

            # Rotation
            rotation_format = config.get_string("rotation/format", "euler")
            value = config.get_vector3d("rotation/value", [0, 0, 0])
            if rotation_format == "euler":
                # Rotation, specified as euler angles
                rotation_euler = MathUtility.transform_point_to_blender_coord_frame(value, self.source_frame)
            elif rotation_format == "forward_vec":
                # Rotation, specified as forward vector
                forward_vec = Vector(MathUtility.transform_point_to_blender_coord_frame(value, self.source_frame))
                # Convert forward vector to euler angle (Assume Up = Z)
                rotation_euler = forward_vec.to_track_quat('-Z', 'Y').to_euler()
            elif rotation_format == "look_at":
                # Compute forward vector
                forward_vec = value - position
                forward_vec.normalize()
                # Convert forward vector to euler angle (Assume Up = Z)
                rotation_euler = forward_vec.to_track_quat('-Z', 'Y').to_euler()
            else:
                raise Exception("No such rotation format:" + str(rotation_format))

            rotation_matrix = Euler(rotation_euler, 'XYZ').to_matrix()

            if rotation_format == "look_at" or rotation_format == "forward_vec":
                inplane_rot = config.get_float("rotation/inplane_rot", 0.0)
                rotation_matrix = rotation_matrix @ Euler((0.0, 0.0, inplane_rot)).to_matrix()

            obj2world_matrix = Matrix.Translation(Vector(position)) @ rotation_matrix.to_4x4()
        else:
            obj2world_matrix = Matrix(np.array(config.get_list("obj2world_matrix")).reshape(4, 4).astype(np.float32))
        return obj2world_matrix


