pkill -9 -f train_emb.py
nohup python train_emb.py --use_wandb &
> nohup.out
tail -f nohup.out