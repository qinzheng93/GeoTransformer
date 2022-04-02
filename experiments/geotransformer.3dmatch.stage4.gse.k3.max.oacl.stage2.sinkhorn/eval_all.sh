for n in $(seq 20 40); do
    python test.py --test_epoch=$n --benchmark=$1 --verbose
    python eval.py --test_epoch=$n --benchmark=$1 --method=lgr
done
# for n in 250 500 1000 2500; do
#     python eval.py --test_epoch=$1 --num_corr=$n --run_matching --run_registration --benchmark=$2
# done
