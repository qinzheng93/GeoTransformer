if [ "$2" = "test" ]; then
    python test.py --test_epoch=$1
fi
python eval.py --test_epoch=$1 --method=lgr
