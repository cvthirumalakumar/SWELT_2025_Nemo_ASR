NEMO_ROOT=/sd1/kumar/code/swelt_2025/Nemo_ASR/NeMo
model_path='/sd1/kumar/code/swelt_2025/Nemo_ASR/pre_trained_model/Conformer-CTC-BPE-Large_IndianEnglish.nemo'
dataset_manifest='/sd1/kumar/code/swelt_2025/Nemo_ASR/data/sampled_data/test/test_manifest.json'
kenlm_model='/sd1/kumar/code/swelt_2025/Nemo_ASR/lm/lm_5.bin'
preds_file='nemo_preds_lm_5'
alpha=2.0
beta=-1.0
beam=128



CUDA_VISIBLE_DEVICES=0 python $NEMO_ROOT/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py nemo_model_file=${model_path} \
           input_manifest=${dataset_manifest} \
           kenlm_model_file=${kenlm_model} \
           beam_width=[${beam}] \
           beam_alpha=[2.0] \
           beam_beta=[-1.0] \
           preds_output_folder=${preds_file} \
           decoding_mode=beamsearch_ngram
