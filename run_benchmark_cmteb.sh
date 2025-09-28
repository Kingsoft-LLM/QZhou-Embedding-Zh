normalize=true
use_instruction=true
export TOKENIZERS_PARALLELISM=true
embed_dim=1792 # 128, 256, 512, 768, 1024, 1280, 1536, 1792

model_name_or_path=Kingsoft-LLM/QZhou-Embedding-Zh

python3 ./run_cmteb_all.py \
    --model_name_or_path ${model_name_or_path}  \
    --normalize ${normalize} \
    --dim ${embed_dim} \
    --use_instruction ${use_instruction} \
    --output_dir <output dir>