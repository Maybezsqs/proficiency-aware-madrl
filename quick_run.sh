#!/bin/bash

#python maddpg/experiments/train.py --display --load-dir ~/results/1/3v1_50_35k_1k_200-3-12-11-11-36-750000 > maddpg1.log

#python maddpg/experiments/train.py --display --load-dir ~/results/2/3v1_50_35k_1k_200-goodddpg-3-12-14-5-32-1200000 --good-policy ddpg > goodddpg1.log

#python maddpg/experiments/train.py --display --load-dir ~/results/3/3v1_50_35k_1k_200-ddpg-3-12-17-19-56-200000 --good-policy ddpg --adv-policy ddpg > ddpg1.log

python maddpg/experiments/train.py --exp-name improve-2_3


python maddpg/experiments/train.py --exp-name improve-2-ddpg-2 --good-policy ddpg --adv-policy ddpg

python maddpg/experiments/train.py --exp-name improve-2-ddpg-3 --good-policy ddpg --adv-policy ddpg

python maddpg/experiments/train.py --exp-name improve-2_4
python maddpg/experiments/train.py --exp-name improve-2_5

python maddpg/experiments/train.py --exp-name improve-2-ddpg-4 --good-policy ddpg --adv-policy ddpg

python maddpg/experiments/train.py --exp-name improve-2-ddpg-5 --good-policy ddpg --adv-policy ddpg
