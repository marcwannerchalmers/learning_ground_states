#!/bin/bash

startids=(65 329 575 842 1100 1355 1615 1860 2122 2383 2633 2890 3129 3405 3654 3917)

let npoints_orig=64
let npoints=$npoints_orig
let offset=0
let startid_idx=$1/3
let offset=($1%3)*$npoints
echo $startid_idx
echo $offset
let startid=${startids[$startid_idx]}+$offset
echo $startid

:'for i in $(seq 0 47);
do
    startid_idx=$i/$steps
    offset=($i%$steps)*$npoints
    declare -i startid=${startids[$startid_idx]}+$offset
    echo $startid
    # echo $offset
done
'


