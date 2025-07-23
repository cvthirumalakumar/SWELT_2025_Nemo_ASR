# SWELT\_2025\_NeMo\_ASR

This repository contains practical scripts and steps for **training Automatic Speech Recognition (ASR) models using NVIDIA NeMo** as part of the **SWELT 2025 Workshop**.

---

## **Steps to Train ASR**

### **1️⃣ Data Preapartion**
Prepare your data by formatting in **NeMo manifest format**:

```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 3.45, "text": "your transcript here1"}
{"audio_filepath": "/path/to/audio1.wav", "duration": 3.45, "text": "your transcript here2"}
```

### **2️⃣ Train Tokenizer**

If you choose **BPE (Byte-Pair Encoding) as your output units** (recommended if your data size is sufficient):

```bash
./data/train_tokenizer.sh
```

This will train a SentencePiece BPE tokenizer on your transcriptions and generate a tokenizer model and vocabulary for use during ASR training.


### **3️⃣ Update Config with Dataset and Tokenizer**

Edit your `conf/conformer*.yaml` to:

* Add paths to your **training, validation, and test manifest files**.
* Add the **tokenizer vocabulary path or directory**.
* Adjust **batch size, number of epochs, and device settings** if needed.


### **4️⃣ Start ASR Training**

Run the training script to start training your ASR model:

```bash
python speech_to_text_ctc_bpe.py
```



### **5️⃣ Evaluate the Trained ASR Model**

After training, evaluate your model on your test dataset:

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.restore_from("path/to/trained/model")
transcript = asr_model.transcribe(["path/to/audio_file.wav"])[0].text
```


---
## **Langauge Model Integration**

### **1️⃣ Training N-Gram LM**

Change arguments in `lm/train_lm.sh` and run the following script to train LM

```bash 
./lm/train_lm.sh
```

### **2️⃣ Decoding with N-Gram LM**
Change arguments in `lm/infer_with_lm.sh` and run the following script to infer with LM

```bash 
./lm/infer_with_lm.sh
```


## **Force-Alignment**
Follow README.md in `force-align` folder for force-aligning using CTC ASR

### **1️⃣ Force-Alignment with CTC ASR**

Change arguments such as model_path, manifest, blank_idx etc in `force-align/force-align.py` and run the following script to generate time stamps file

```bash 
python force-align/force-align.py
```
The above script will run for only one file and the time stamps will be same in `temp_time_stamps.txt` file. You can change the script to do it for all files.


🚀 Happy learning and hands-on ASR training with NeMo!
