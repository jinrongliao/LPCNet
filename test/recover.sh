#!/bin/sh
python3 run_plc_experiment.py
./../lpcnet_demo -plc_file causal plc_pattern.txt output_dropped.pcm output.pcm
