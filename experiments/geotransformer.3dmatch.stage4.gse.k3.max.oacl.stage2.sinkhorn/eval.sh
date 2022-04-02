if [ "$3" = "test" ]; then
    python test.py --test_epoch=$1 --benchmark=$2
fi
python eval.py --test_epoch=$1 --benchmark=$2 --method=lgr
# for n in 250 500 1000 2500; do
#     python eval.py --test_epoch=$1 --num_corr=$n --run_matching --run_registration --benchmark=$2
# done
