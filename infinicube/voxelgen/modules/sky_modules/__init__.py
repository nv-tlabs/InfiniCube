from .skybox_mlp_modulator import SkyboxMlpModulator
from .skybox_null import SkyboxNull
from .skybox_panorama_full import SkyboxPanoramaFull


def convert_to_camel_case(string):
    return "".join(word.capitalize() for word in string.split("_"))


__all__ = [
    "SkyboxPanoramaFull",
    "SkyboxMlpModulator",
    "SkyboxNull",
    "convert_to_camel_case",
]
