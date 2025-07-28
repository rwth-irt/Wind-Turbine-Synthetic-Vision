# Wind Turbine Synthetic Vision
# Copyright (C) 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders
# Institute of Automatic Control - RWTH Aachen University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import blenderproc as bproc
import os
import sys


sys.path.insert(0, "lib")
try:
    import helper as helper
    from generator_200 import DatasetGenerator, Parameter
except ImportError:
    print('No Import')


output_path_abs, output_paths = helper.get_output_paths("random_noise", os.path.dirname(__file__))

generator = DatasetGenerator(
    Parameter,
    10,
    os.path.join(os.path.dirname(__file__), "./scene/WELT_yolo.blend"),
    output_paths,
    os.path.join(os.path.dirname(__file__),"background"),
    output_path_abs
)

generator.generate()
