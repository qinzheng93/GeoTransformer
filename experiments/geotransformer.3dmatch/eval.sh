python test.py --test_epoch=$1 --benchmark=$2 --verbose
python eval.py --test_epoch=$1 --run_matching --run_registration --benchmark=$2
