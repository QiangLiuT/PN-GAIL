# PN-GAIL: Leveraging Non-optimal Information from Imperfect Demonstrations (ICLR 2025)
This repository contains the PyTorch code for the paper "PN-GAIL: Leveraging Non-optimal Information from Imperfect Demonstrations"

## Requirement
 * Python 3.9.18
 * PyTorch 2.0.1
 * gym 0.25.2
 * mujoco
 * numpy 1.26.2

## Execute
 * PN_GAIL
 ```
 python PN_GAIL.py --env Ant-v2 --num-epochs 6000 --traj-size 600 --save 
 ```
 * 2IWIL
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 6000 --traj-size 600 --weight --save
 ```
 * IC_GAIL
 ```
 python IC_GAIL.py --env Ant-v2 --num-epochs 6000 --traj-size 600 --save
 ```
 * GAIL
 ```
 python 2IWIL.py --env Ant-v2 --num-epochs 6000 --traj-size 600
 ```
 * WGAIL
 ```
 python wgail.py --env Ant-v2 --num-epochs 6000 --traj-size 600 --save
 ```


## Acknowledegement
We would like to thank the authors of [2IWIL/IC-GAIL](https://github.com/kristery/Imitation-Learning-from-Imperfect-Demonstration). Our code structure is largely based on their source code.
