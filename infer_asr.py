import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")
transcript = asr_model.transcribe(["path/to/audio_file.wav"])[0].text