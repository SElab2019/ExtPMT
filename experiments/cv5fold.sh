#!/bin/bash

# 11 for 40 project, 22 for 654 projects
Work_Mode=22

for fld in 1 2 3 4 5
do
	#file movement
	python PATH_TO_/experiments/cv5fold_file_copy.py $Work_Mode $fld
	#traing & testing
	python PATH_TO_/models/m_CNN.py
done