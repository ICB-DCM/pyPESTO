"""Constants related to visualization methods."""

from typing import Tuple, Union

LEN_RGB = 3  # number of elements in an RGB color
LEN_RGBA = 4  # number of elements in an RGBA color
RGB = Tuple[(float,) * LEN_RGB]  # typing of an RGB color
RGBA = Tuple[(float,) * LEN_RGBA]  # typing of an RGBA color
RGB_RGBA = Union[RGB, RGBA]  # typing of an RGB or RGBA color
RGBA_MIN = 0  # min value for an RGBA element
RGBA_MAX = 1  # max value for an RGBA element
RGBA_ALPHA = 3  # zero-indexed fourth element in RGBA
RGBA_WHITE = (RGBA_MAX, RGBA_MAX, RGBA_MAX, RGBA_MAX)  # white as an RGBA color
RGBA_BLACK = (RGBA_MIN, RGBA_MIN, RGBA_MIN, RGBA_MAX)  # black as an RGBA color
