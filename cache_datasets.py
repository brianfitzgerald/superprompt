from datasets import load_dataset

dataset = load_dataset("roborovski/diffusiondb-seq2seq")
dataset = load_dataset("THUDM/ImageRewardDB", "4k", verification_mode="no_checks")
dataset = load_dataset("bentrevett/multi30k")