#!/bin/bash

echo 'Experiments'
echo '0: ParaLiNGAM - Real Dataset'
echo '1: ParaLiNGAM - Synthetic'
echo '2: ParaLiNGAM - Impact of threshold mechanism'
echo '3: ParaLiNGAM - Vs Baseline methods'
echo '4: DirectLiNGAM (cpu) - Real Dataset'
echo '5: DirectLiNGAM (cpu) - Synthetic'
echo '6: DirectLiNGAM Opt (cpu) - Real Dataset'
echo '7: DirectLiNGAM Opt (cpu) - Synthetic'

printf "Please select the experiment [0-7]: "
read algorithm_id


if [[ $algorithm_id -le 3 ]]; then
	cd GPU
    make
    ./ParaLiNGAM $algorithm_id
fi

if [[ $algorithm_id -gt 3 ]]; then
	cd CPU
    make
    ./LiNGAM $algorithm_id
fi

