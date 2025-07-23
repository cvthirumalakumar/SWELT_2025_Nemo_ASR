import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.restore_from("path/to/trained/model")
transcript = asr_model.transcribe(["path/to/audio_file.wav"])[0].text
