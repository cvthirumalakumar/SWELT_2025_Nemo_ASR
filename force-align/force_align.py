import nemo.collections.asr as nemo_asr
import torch
import torchaudio.functional as F
import glob
import json
from tqdm import tqdm
import os
import torchaudio

def read_text(file_path):
    with open(file_path,'r',encoding='utf8') as f:
        text = [l.strip() for l in f.readlines()]
    return text

def read_jsonl(json_path):
    data_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list

def align(emission, tokens, blank_idx):
    targets = torch.tensor([tokens], dtype=torch.int32)
    alignments, scores = F.forced_align(emission, targets, blank=blank_idx)
    alignments, scores = alignments[0], scores[0]  
    scores = scores.exp()
    return alignments, scores

def get_time_stamps_with_probs(emission, token_ids, blank_idx, tokenizer, frame_size):
    aligned_tokens, alignment_scores = align(emission, token_ids, blank_idx)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores, blank=blank_idx)
    words = {}
    current_start_idx = 0
    word_tokens = []
    word_token_probs = []
    for idx,s in enumerate(token_spans):
        token_txt = tokenizer.ids_to_tokens([int(s.token)])[0]
        if token_txt[0]=='▁' or idx == len(token_spans)-1: 
            if idx == len(token_spans)-1:
                word_tokens.append(token_txt)
                word_token_probs.append(float(s.score))
            if len(word_tokens)!=0:
                prev_completed_word = "".join(word_tokens)
                words[prev_completed_word] = {'score': sum(word_token_probs)/len(word_token_probs),
                                              'start_idx': current_start_idx,
                                              'end_idx': s.end,
                                              'start_time': current_start_idx*frame_size,
                                              'end_time':  s.end*frame_size
                                              }
            current_start_idx = s.end
            word_tokens = [token_txt]
            word_token_probs = [float(s.score)]
        else:
            word_tokens.append(token_txt)
            word_token_probs.append(float(s.score))
    return words

def main():

    model_path = '/sd1/kumar/code/swelt_2025/Nemo_ASR/pre_trained_model/Conformer-CTC-BPE-Large_IndianEnglish.nemo'
    asr_model = nemo_asr.models.ASRModel.restore_from(
                restore_path=model_path
            )
    asr_model.eval()
    blank_idx = 128  ### Blank index is num_classes for this model it is 128
    wav_paths = []
    transcripts = []
    manifest = "/sd1/kumar/code/swelt_2025/Nemo_ASR/data/sampled_data/test/test_manifest.json"
    for sample in read_jsonl(manifest):
        wav_paths.append(sample['audio_filepath'])
        transcripts.append(sample['text'])
        break

    
    print("Extracting posterior logprobabilities...")
    waveform, sr = torchaudio.load(wav_paths[0])
    print(waveform.shape, sr)
    emissions = asr_model.transcribe(wav_paths, return_hypotheses=True)
    print(emissions[0].y_sequence.shape)
    n_samples_per_frame = waveform.shape[1]/emissions[0].y_sequence.shape[0]
    frame_size = n_samples_per_frame/sr
    print("Force Aligning....")
    for emission, wav_id, transcript in tqdm(zip(emissions, wav_paths, transcripts), total=len(wav_paths)):
        emission = torch.tensor(emission.y_sequence).unsqueeze(0)
        tokens_txt = asr_model.tokenizer.text_to_tokens(transcript)
        token_ids = asr_model.tokenizer.tokens_to_ids(tokens_txt)
        aligned_tokens, alignment_scores = align(emission, token_ids, blank_idx)
        print(aligned_tokens)
        token_spans = F.merge_tokens(aligned_tokens, alignment_scores, blank=blank_idx)
        print(token_spans)
        tokens_text = [asr_model.tokenizer.ids_to_tokens([int(s.token) for s in token_spans])]
        print(tokens_text)
        words = get_time_stamps_with_probs(emission, token_ids, blank_idx, asr_model.tokenizer, frame_size)
        print(words)
        with open("temp_time_stamps.txt",'w',encoding='utf8') as f:
            for word in words:
                word_str = word
                word = words[word]
                f.write(str(word['start_time'])+"\t"+str(word['end_time'])+"\t"+word_str+"\n")

        break


if __name__ == '__main__':
    main()











    