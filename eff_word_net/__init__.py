"""
.. include:: ../README.md
"""

import os
RATE=16000
samples_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)),"sample_refs")

from eff_word_net.package_installation_scripts import check_install_tflite

check_install_tflite()
