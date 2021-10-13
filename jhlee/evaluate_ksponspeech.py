import torch
from speechbrain.pretrained import EncoderDecoderASR
import csv
import time
import jiwer
import argparse

class SpeechBrainInferencer(object):
    def __init__(
        self,
        source: str = "ddwkim/asr-conformer-transformerlm-ksponspeech",
        savedir: str = "/data/voice-part/jhlee/saltlux/asr/speechbrain_korean/speechbrain/jhlee/pretrained_model/conformer",
        use_cuda: bool = True,
    ):
        if torch.cuda.is_available() and use_cuda:
            run_opts = {"device": "cuda"}
        else:
            run_opts = None
        self.model = EncoderDecoderASR.from_hparams(
            source=source,
            savedir=savedir,
            run_opts=run_opts,
        )

    def __call__(self, audio: str):
        transcript = self.model.transcribe_file(audio)
        return transcript


def save_csv(hypothesis, ground_truth, path="result.csv"):
    f = open(path, "w", newline="")
    wr = csv.writer(f, delimiter=' ')

    for i in range(len(ground_truth)):
        wr.writerows([ground_truth[i], hypothesis[i]])

    f.close()


if __name__ == "__main__":
    EVAL_CLEAN = '/data/STT_data/KsponSpeech-mini/eval_clean_transcript.txt'
    EVAL_OTHER = '/data/STT_data/KsponSpeech-mini/eval_other_transcript.txt'
    TEST = '/data/STT_data/KsponSpeech-mini/test_transcript.txt'

    EVAL_CLEAN_WAV = '/data/STT_data/KsponSpeech-mini/eval_clean_wav/'
    EVAL_OTHER_WAV = '/data/STT_data/KsponSpeech-mini/eval_other_wav/'
    TEST_WAV = '/data/STT_data/KsponSpeech-mini/test_wav/'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--split", type=str, default="eval_clean")
    config = parser.parse_args()
    print(config)
    if config.split == "eval_clean":
        script = EVAL_CLEAN
        wav_dir = EVAL_CLEAN_WAV

    elif config.split == "eval_other":
        script = EVAL_OTHER
        wav_dir = EVAL_OTHER_WAV
    
    elif config.split == "test":
        script = TEST
        wav_dir = TEST_WAV

    inferencer = SpeechBrainInferencer(
        source="ddwkim/asr-conformer-transformerlm-ksponspeech",
        savedir="/data/voice-part/jhlee/saltlux/asr/speechbrain_korean/speechbrain/jhlee/pretrained_model/conformer",
    )

    ground_truth = []
    hypothesis = []

    with open(script) as f:
        lines = f.readlines()
        start = time.time()
        for line in lines:
            line = line.strip()
            wav = wav_dir + line[:18] + ".wav"
            ground_truth.append(line[19:])
            hypothesis.append(inferencer(wav))

    print("time: ", time.time() - start)
    print("WER: ", jiwer.wer(ground_truth, hypothesis))

    f1 = open('ground_truth.txt', 'w')  
    for i in range(len(ground_truth)):
        f1.write(ground_truth[i]+'\n')
    f1.close()

    f2 = open('hypothesis.txt', 'w')  
    for i in range(len(hypothesis)):
        f2.write(hypothesis[i]+'\n')
    f2.close()

    print("complete")
