import nemo.collections.asr as nemo_asr
import torch
import torchaudio.functional as F
import glob
import json
from tqdm import tqdm
import os

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


def get_words_with_probs(emission, token_ids, blank_idx, tokenizer):
    targets = torch.tensor([tokens], dtype=torch.int32)
    alignments, scores = F.forced_align(emission, targets, blank=blank_idx)

    alignments, scores = alignments[0], scores[0] 
    aligned_tokens, alignment_scores = align(emission, token_ids, blank_idx)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores, blank=blank_idx)
    words = []
    word_probs = []
    for idx,s in enumerate(token_spans):
        token_txt = tokenizer.ids_to_tokens([int(s.token)])[0]
        if token_txt[0]=='▁':
            if idx!=0:
                prev_completed_word = "".join(word_tokens)
                words.append(prev_completed_word)
                word_probs.append(sum(word_token_probs)/len(word_token_probs))
            if token_txt == '▁':
                word_tokens = []
                word_token_probs = []
            elif len(token_txt)>1:
                word_tokens = [token_txt[1:]]
                word_token_probs = [float(s.score)]
        else:
            word_tokens.append(token_txt)
            word_token_probs.append(float(s.score))

        if idx==len(token_spans)-1:
            prev_completed_word = "".join(word_tokens)
            words.append(prev_completed_word)
            word_probs.append(sum(word_token_probs)/len(word_token_probs))
    return words, word_probs

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
    emissions = asr_model.transcribe(wav_paths, num_workers=4, logprobs=True)

    print("Force Aligning....")
    problems = []
    final_dict = {}
    for emission, wav_id, transcript in tqdm(zip(emissions, wav_paths, transcripts), total=len(wav_paths)):
        emission = torch.tensor(emission).unsqueeze(0)
        tokens_txt = asr_model.tokenizer.text_to_tokens(transcript)
        token_ids = asr_model.tokenizer.tokens_to_ids(tokens_txt)
        try:
            words, word_probs = get_words_with_probs(emission, token_ids, blank_idx, asr_model.tokenizer)
            
            final_dict[wav_id] = {
                'transcript':transcript,
                'words':words,
                'word_probs':word_probs
            }
        except:
            problems.append(wav_id)
        # print("word lebel probabilities")
        # for word,score in zip(words,word_probs):
        #     print(f"{word}:{score}")

    out_folder = "fa-likelihoods"
    os.makedirs(out_folder,exist_ok=True)
    with open(out_folder+"/imprint_vakyansh-conformer-ctc.json", 'w') as json_file:
        json.dump(final_dict, json_file, indent=4)
    print(problems)

if __name__ == '__main__':
    main()











    