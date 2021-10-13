import torch
from speechbrain.pretrained import EncoderDecoderASR
import csv
import time
import jiwer


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
    wr = csv.writer(f, delimiter=' ', quotechar='', quoting=csv.QUOTE_MINIMAL)

    for i in range(len(ground_truth)):
        wr.writerows([ground_truth[i], hypothesis[i]])

    f.close()


if __name__ == "__main__":
    inferencer = SpeechBrainInferencer(
        source="ddwkim/asr-conformer-transformerlm-ksponspeech",
        savedir="pretrained_model/conformer",
    )

    EVAL_CLEAN = '/data/STT_data/KsponSpeech-mini/eval_clean_transcript.txt'
    EVAL_OTHER = '/data/STT_data/KsponSpeech-mini/eval_other_transcript.txt'
    TEST = '/data/STT_data/KsponSpeech-mini/test_transcript.txt'

    EVAL_CLEAN_WAV = '/data/STT_data/KsponSpeech-mini/eval_clean_wav/'
    EVAL_OTHER_WAV = '/data/STT_data/KsponSpeech-mini/eval_other_wav/'
    TEST_WAV = '/data/STT_data/KsponSpeech-mini/test_wav/'

    script = TEST
    wav_dir = TEST_WAV

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
    save_csv(hypothesis, ground_truth, "reslut.csv")
    print("complete")
