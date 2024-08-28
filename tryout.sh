#!/bin/bash

seqs=("rand" "lds")
splits=(0.1 0.3 0.5 0.7 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9)
Lxs=(9 9 9 9 9 9 9 9 9 9)
delta1s=(0 0 0 0 0 0 1 2 3 4 5 0 0 0 0 0)

python -u train_regression.py --seq ${seqs[$1%2]} --test-size ${splits[$1%15]} --nrow ${Lxs[$1%15]} --delta1 ${delta1s[$1%15]}




