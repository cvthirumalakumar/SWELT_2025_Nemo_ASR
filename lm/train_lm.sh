# train_file='/mnt/sd1/kumar/data/librispeech/librispeech-lm-norm.txt'
NEMO_ROOT=/sd1/kumar/code/swelt_2025/Nemo_ASR/NeMo
train_file='/sd1/kumar/code/swelt_2025/Nemo_ASR/data/sampled_data/train/train_manifest.json'
model='/sd1/kumar/code/swelt_2025/Nemo_ASR/pre_trained_model/Conformer-CTC-BPE-Large_IndianEnglish.nemo'
output='lm_5.bin'
kenlm_bin_path=/sd1/kumar/code/swelt_2025/Nemo_ASR/NeMo/decoders/kenlm/build/bin

python $NEMO_ROOT/scripts/asr_language_modeling/ngram_lm/train_kenlm.py train_paths=[${train_file}] \
    nemo_model_file=${model} \
    kenlm_model_file=${output} \
	ngram_length=5 \
	kenlm_bin_path=${kenlm_bin_path} 
