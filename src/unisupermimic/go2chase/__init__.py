"""Unitree Go2 chase / dodge with SuperMimic, ported from go2-supermimic.

The algorithm (``supermimic/``: world model, imagination, trainer) and the target
dynamics (``target/``) are go2-supermimic's (MIT, Ziyue Zheng; see LICENSE_go2_supermimic),
reformatted, with imports moved under ``unisupermimic.go2chase``. The environment
(``env/``) is rewritten for UniLab; ``supermimic/state.py`` reads the robot through it.
"""
