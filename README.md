# FastSpeech2
This is my implementation and review of text-to-speech system [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://github.com/xcmyz/FastSpeech). This project is based on [xcmyz's](https://github.com/xcmyz/FastSpeech) and [mingo24](https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3) implementations of FastSpeech and FastSpeech2. 
Feel free to use/modify the code.


# How to run?

[mingo24](https://github.com/ming024/FastSpeech2/tree/d4e79eb52e8b01d24703b2dfc0385544092958f3) advise to prepare data by downloading it on your own, put it on a special directory, and only after that start preprocessing and training

1) download datasets and put them in output/ckpt/LJSpeech/, output/ckpt/AISHELL3, or output/ckpt/LibriTTS/
2) install the Python dependencies
3) run synthesis.py
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
