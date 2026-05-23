# # 下载 checkpoint
# bcecmd bos cp -r bos:/nlp-data-app-models/huanghj/MLLM-MSR/checkpoints/sft_microlens/epoch_2/ ./checkpoints/sft_microlens/epoch_2/

# 评测 SFT
python quick_eval_sft.py --checkpoint_path ./checkpoints/sft_microlens/epoch_0 --dataset microlens --num_users 100

# 评测 base model（去掉 PeftModel.from_pretrained 那行即可）
