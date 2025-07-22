NEMO_ROOT=/sd1/kumar/code/swelt_2025/Nemo_ASR/NeMo
python $NEMO_ROOT/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest="/sd1/kumar/code/swelt_2025/Nemo_ASR/data/sampled_data/train/train_manifest.json" \
        --data_root="/sd1/kumar/code/swelt_2025/Nemo_ASR/data/tokenizer" \
        --vocab_size=128 \
        --tokenizer="spe" \
        --spe_type="unigram" \
        --spe_character_coverage=1.0 \
        --log