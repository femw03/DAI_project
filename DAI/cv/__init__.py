"""A Module that contains our implementation of the ComputerVisionModule"""

import logging

from .cv import ComputerVisionModuleImp  # noqa: F401

logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
