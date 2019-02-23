python3 train.py -c configs/bbox1.cfg > /dev/null 2>&1 &
python3 eval.py -c configs/bbox1.cfg > /dev/null 2>&1

python3 train.py -c configs/bbox2.cfg > /dev/null 2>&1 &
python3 eval.py -c configs/bbox2.cfg > /dev/null 2>&1

python3 train.py -c configs/bbox3.cfg > /dev/null 2>&1 &
python3 eval.py -c configs/bbox3.cfg > /dev/null 2>&1
