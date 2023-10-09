fn_name="train_ir"
pkill -9 -f $fn_name.py
nohup python $fn_name.py --use_wandb &
> nohup.out
tail -f nohup.out