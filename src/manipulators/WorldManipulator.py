import bpy

from src.main.Module import Module
from src.utility.BlenderUtility import get_all_mesh_objects


class WorldManipulator(Module):
    """ Allows manipulation of the current World in the scene via specifying one or more {attr name/custom prop. name/
        custom function name: value} pairs.

        Example 1: Sets the World's custom property `category_id` to 123.

        {
          "module": "manipulators.WorldManipulator",
          "config": {
            "cp_category_id": 123
          }
        }

        Example 2: Sets the color and the strength of the light emitted by the background surface.

        {
          "module": "manipulators.WorldManipulator",
          "config": {
            "cf_bg_surface_color": [1, 1, 1, 1],
            "cf_bg_surface_strength": 100
          }
        }

        Example 3: Disables shader node tree of the background surface and sets a solid color to the background surface.

        {
          "module": "manipulators.WorldManipulator",
          "config": {
            "use_nodes": False,
            "color": [0.5, 0.5, 0.5]
          }
        }

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "key": "Name of the attribute/custom property to change or a name of a custom function to perform on objects. "
               "Type: string. "
               "In order to specify, what exactly one wants to modify (e.g. attribute, custom property, etc.): "
               "For attribute: key of the pair must be a valid attribute name of the world. "
               "For custom property: key of the pair must start with `cp_` prefix. "
               "For calling custom function: key of the pair must start with `cf_` prefix. See table below for "
               "supported custom function names."
        "value": "Value of the attribute/custom prop. to set or input value(s) for a custom function. Type: string, "
                 "int, bool or float, list/Vector."

    **Custom functions**:

    .. csv-table::
        :header: "Parameter", "Description"

        "cf_bg_surface_color", "Sets the RGBA color of the light emitted by the background. Type: mathutils.Vector."
        "cf_bg_surface_strength", "Sets the strength of the light emitted by the background. Type: float."
        "cf_set_world_category_id", "Sets the category_id of the background. Type: int."
    """

    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        """ Assigns user-defined values to World's attributes, custom properties, or manipulates the state of the world.
            1. Selects current active World.
            2. Change World's state via setting user-defined values to it's attributes, custom properties, etc.
        """
        world = bpy.context.scene.world
        for key in self.config.data.keys():
            requested_cp = False
            requested_cf = False

            value = self.config.get_raw_value(key)

            if key.startswith('cp_'):
                requested_cp = True
                key = key[3:]
            elif key.startswith('cf_'):
                requested_cf = True
                key = key[3:]
            if hasattr(world, key) and all([not requested_cp, not requested_cf]):
                setattr(world, key, value)
            elif requested_cp:
                world[key] = value
            elif requested_cf:

                if key == "bg_surface_color":
                    self._set_bg_surface_color(world, value)
                elif key == "bg_surface_strength":
                    self._set_bg_surface_strength(world, value)
                elif key == "set_world_category_id":
                    if isinstance(value, int):
                        bpy.context.scene.world["category_id"] = value
                    else:
                        raise Exception("The category id of the world can only be int!")
                else:
                    raise RuntimeError('Unknown cf_ parameter: ' + key)

            else:
                raise RuntimeError('Unknown parameter: ' + key)

    def _set_bg_surface_color(self, world, color):
        """ Sets the color of the emitted light by the background surface.

        :param world: World to modify. Type: bpy.types.World.
        :param color: RGBA color of the emitted light. Type: mathutils.Vector.
        """
        import os
        if color == "sky_texture":
            sky_tex = world.node_tree.nodes.new(type='ShaderNodeTexSky')
            sky_tex.sky_type = "HOSEK_WILKIE"
            sky_tex.ground_albedo = 0.9
            sky_tex.turbidity = 2.2

            back_node = world.node_tree.nodes['Background']
            world.node_tree.links.new(sky_tex.outputs['Color'], back_node.inputs['Color'])
        
        elif os.path.isdir(color):

            random_image = self.random_image_from_folder(color)
            img_name = os.path.basename(random_image)
            bpy.data.images.load(random_image)

            env_node = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
            back_node = world.node_tree.nodes['Background']
            world.node_tree.links.new(env_node.outputs['Color'], back_node.inputs['Color'])

            end_frame = bpy.context.scene.frame_end

            env_node.image = bpy.data.images[img_name]
            env_node.image.source = "SEQUENCE"

            env_node.image_user.frame_duration = end_frame
            env_node.image_user.frame_start = 0
            env_node.image_user.frame_offset = 0
            env_node.image_user.use_cyclic = True
            env_node.image_user.use_auto_refresh = True
            
            
        else:
            world.node_tree.nodes["Background"].inputs['Color'].default_value = color

    def _set_bg_surface_strength(self, world, strength):
        """ Sets the strength of the emitted light by the background surface.

        :param world: World to modify. Type: bpy.types.World.
        :param strength: Strength of the emitted light. Type: float.
        """
        world.node_tree.nodes["Background"].inputs['Strength'].default_value = strength
    

    def random_image_from_folder(self, folder_name):
        """ Get random image from folder that act as world color

        :param world: World to modify. Type: bpy.types.World.
        :param folder_name: Folder where images should be taken 
        """
        import os
        import random

        print("This method will change all the image name inside folder, proceed with caution")

        all_image_path = []

        bg_list = os.listdir(folder_name)
        random.shuffle(bg_list)

        for i, files in enumerate(bg_list):
            all_image_path.append(os.path.join(folder_name, files))
            src = os.path.join(folder_name, files)
            dst = os.path.join(folder_name, f"temp_bg_{str(i).zfill(10)}.jpg")
            
            os.rename(src, dst)

        bg_list = os.listdir(folder_name)
        random.shuffle(bg_list)

        for i, files in enumerate(bg_list):
            all_image_path.append(os.path.join(folder_name, files))
            src = os.path.join(folder_name, files)
            dst = os.path.join(folder_name, f"bg_{str(i).zfill(10)}.jpg")
            
            os.rename(src, dst) 
        
        return dst
        