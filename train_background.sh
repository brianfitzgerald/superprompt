pkill -9 -f train_seq2seq.py
nohup python train_seq2seq.py --use_wandb &
> nohup.out
tail -f nohup.out