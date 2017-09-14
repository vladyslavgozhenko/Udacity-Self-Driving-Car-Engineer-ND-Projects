#!/bin/bash
# Start caclulation model with different parameters
#python prepare_files_XX.py [protation] - generates training data
# python file model.py [angle correction for  left-right camera images],[vert_rotation],[batch size],[nbepoch]

python3 prepare_files.py 0.25
python3 model.py 0.2 0.25 32 3
