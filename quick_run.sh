#!/bin/bash

python mixedcoop/experiments/train.py --display --load-dir ~/results/[SVAED_FILE] --good-policy [POLICY] --adv-policy [POLICY] > [LOG_FILE]
