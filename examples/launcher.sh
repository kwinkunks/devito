#!/bin/bash

# Launch as:
#
#    problem=[acoustic,tti] order=int grid=int mode=[blocked bs=int,original,maxperf,dse,dle] vtune=[...] advisor=[...] cprofile=[...] ./path/to/devito/examples/launcher.sh
#

if [ -z ${DEVITO_HOME+x} ]; then
    echo "Please, set DEVITO_HOME to the root Devito directory"
    exit
fi

if [ -z ${grid+x} ]; then
    grid=256
fi

if [ -z ${order+x} ]; then
    order=4
fi

time_orders="2"
if [ "$problem" == "acoustic" ]; then
    space_orders="2 4 6 8 10 12 14 16"
elif [ "$problem" == "tti" ]; then
    space_orders="4 8"
else
    echo "Unrecognised problem $problem (allowed: acoustic, tti)"
    exit
fi

timestamp=$(date +%Y_%m_%d_%H:%M)
machine=$(hostname)

export OMP_NUM_THREADS=8
export DEVITO_ARCH=intel
export KMP_AFFINITY=explicit,proclist=[0,1,2,3,4,5,6,7]

if [ $OMP_NUM_THREADS == 8 ]; then
    arch="bdwb_ss"
else
    arch="bdwb_ds"
fi

name=$problem-$mode-grid$grid

# Output directories
export DEVITO_RESULTS=$DEVITO_OUTPUT/raw/$machine/$name-$arch-$timestamp
export DEVITO_PLOTS=$DEVITO_OUTPUT/plots/$machine/$name-$arch-$timestamp

if [ -n "$vtune" ]; then
    PREFIX="amplxe-cl -collect $vtune -data-limit=1000 -discard-raw-data -result-dir=$DEVITO_HOME/../profilings/vtune/$name-so$order-$vtune -resume-after=30 -search-dir=/tmp/devito-1000/"
elif [ -n "$advisor" ]; then
    PREFIX="advixe-cl -collect $advisor -data-limit=500 -project-dir=$DEVITO_HOME/../profilings/advisor/$name-so$order-$advisor -resume-after=30000 -search-dir=all:r=/tmp/devito-1000/ -run-pass-thru=--no-altstack"
elif [ -n "$cprofile" ]; then
    SUFFIX="-m cProfile -o profile.dat"
fi

PYTHON="numactl --cpubind=0 --membind=0 python"

if [[ "$mode" == "maxperf" || "$mode" == "dse" || "$mode" == "dle" ]]; then
    benchtype=$mode
    mode="bench"
fi

if [ "$mode" == "bench" ]; then
    $PYTHON $DEVITO_HOME/examples/benchmark.py bench -bm $benchtype -P $problem -a -o -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS
    $PYTHON $DEVITO_HOME/examples/benchmark.py plot -bm $benchtype -P $problem -a -o -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS -p $DEVITO_PLOTS --max_bw 31 --max_flops 537 --point_runtime --arch $arch
elif [ "$mode" == "auto" ]; then
    $PYTHON $DEVITO_HOME/examples/benchmark.py run -P $problem -a -o -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
elif [ "$mode" == "blocked" ]; then
    $PREFIX $PYTHON $SUFFIX $DEVITO_HOME/examples/benchmark.py run -P $problem -cb $bs $bs -o -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
else
    $PREFIX $PYTHON $SUFFIX $DEVITO_HOME/examples/benchmark.py run -P $problem --tn 25 -o -a -d $grid $grid $grid -r $DEVITO_RESULTS -so $order -to 2
fi

if [ -n "$cprofile" ]; then
    gprof2dot -f pstats profile.dat | dot -Tpdf -o profile.pdf
    rm profile.dat
fi
