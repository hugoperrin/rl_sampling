"""Data preparation CLI file."""
import os
from typing import Dict, List, Optional

import fire
import numpy as np
import pyexr
from loguru import logger
from mitsuba_client.client.renderer_enum import RendererEnum
from mitsuba_client.client.rendering_env import RendererEnv
from tqdm import tqdm

from rl_sampling.utils.json_utils import load_json

default_scene_folder: str = os.getenv("DEFAULT_SCENE_FOLDER", os.path.join("scenes"))


class DataPreparationCLI:
    """ Command Line Interface for data preparation functions
    """

    def list_available_scenes(self, scene_folder: Optional[str] = None) -> List[str]:
        """List all available scenes in given folder.

        Args:
            scene_folder (Optional[str], optional): The folder for scenes. Defaults to None.
            In case of None, we use the default path specified through env var

        Returns:
            List[str]: The string list of available scenes.
        """
        if scene_folder is None:
            scene_folder = default_scene_folder
        return list(os.listdir(scene_folder))

    def generate_render_target(
        self,
        config_path: str,
        render_target: str = "low_spp",
        device: str = "gpu_multi",
        mitsuba_variant: Optional[str] = None,
        base_spp_level: int = 4,
    ) -> Dict[str, List[np.ndarray]]:
        """Generate a render target for a given config and render target.

        Args:
            config_path (str): The config path to use
            render_target (str, optional): Which render target to use. Defaults to "low_spp".
            device (str, optional): The device to run it on => renderer type. Defaults to "gpu_multi".
            mitsuba_variant (Optional[str], optional): Overrides the mitsuba variant. Defaults to None.
            base_spp_level (int, optional): The base spp level for each pass in the scene. Defaults to 4.

        Returns:
            Dict[str, List[np.ndarray]]: A dict of all the final renders for each spp levels.
        """
        render_enum: RendererEnum = RendererEnum.from_str(device)
        renderer: RendererEnv = RendererEnv(
            mitsuba_variant=render_enum.default_variant()
            if mitsuba_variant is None
            else mitsuba_variant,
            renderer_type=render_enum,
        )
        config: Dict = load_json(config_path)
        folder_out: Optional[str] = config.get("folder_out")
        os.makedirs(folder_out, exist_ok=True)

        render_targets: Optional[Dict[str, Dict]] = config.get("render_targets")
        if not isinstance(folder_out, str) or render_targets is None:
            raise ValueError(
                "Config is not well formatted, missing folder_out path and/or render_targets",
            )
        target_dict: Optional[Dict[str, int]] = render_targets.get(render_target)
        if target_dict is None:
            raise ValueError(f"There is no target for render with key: {render_target}")
        results_dict: Dict[str, List[np.ndarray]] = {}
        scene_path: str = config.get("scene_path")
        if not isinstance(scene_path, str):
            raise ValueError("There is no proper scene_path specified")
        renderer.load_scene(scene_path=scene_path, scene_id="scene_gen")
        for spp_level, n_renders in tqdm(target_dict.items()):
            logger.info(f"Starting renders for spp level: {spp_level}")
            curr_folder: str = os.path.join(
                folder_out, f"render_{spp_level}",
            )
            os.makedirs(curr_folder, exist_ok=True)
            results_dict[spp_level] = []
            spp_level_int: int = int(spp_level)
            for i in range(n_renders):
                res: List[np.ndarray] = renderer.render(
                    scene_id="scene_gen", n_pass=spp_level_int // base_spp_level,
                )
                results_dict[spp_level].append(res)
                for k, r in enumerate(res):
                    pyexr.write(
                        os.path.join(curr_folder, f"spp_{i}th_{k}.exr",), r,
                    )
        return results_dict


if __name__ == "__main__":
    fire.Fire(DataPreparationCLI)
