import torch
from speechbrain.pretrained import EncoderDecoderASR
import csv
import time
import jiwer


class SpeechBrainInferencer(object):
    def __init__(self, source: str="speechbrain/asr-transformer-transformerlm-librispeech", savedir: str="pretrained_model/transformer", use_cuda: bool=True):
        if torch.cuda.is_available() and use_cuda:
            run_opts={"device": "cuda"}
        else:
            run_opts=None
        self.model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_model/transformer", run_opts=run_opts)


    def __call__(self, audio: str):
        transcript = self.model.transcribe_file(audio)
        return transcript 


def inference(file_path = "/data/voice-part/jhlee/saltlux/asr/speechbrain/samples/audio_samples/example1.wav"):
    inferencer = SpeechBrainInferencer(source = "speechbrain/asr-crdnn-rnnlm-librispeech", str = "pretrained_model/transformer")
    print(inferencer(file_path))


def save_csv(hypothesis, path ='result.csv'):
    f = open('jhlee_upper', 'r')
    rdr = csv.reader(f)
    next(rdr)
    lines = []

    for i, line in enumerate(rdr):
        line.append(hypothesis[i])
        lines.append(line)
 
    f = open(path, 'w', newline='')
    wr = csv.writer(f)
    wr.writerows(["file_path", "file_size", "ground_truth", "hypothesis"])
    wr.writerows(lines)

    f.close()
    

if __name__ == "__main__":
    inferencer = SpeechBrainInferencer(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_model/transformer")
 
    data = open('jhlee_upper.csv')
    reader = csv.reader(data)
    lines = list(reader)

    ground_truth = []
    hypothesis = []

    start = time.time()
    
    for i in range(1, len(lines)):
        ground_truth.append(lines[i][2])
        hypothesis.append(inferencer(lines[i][0]))

    print("time: ", time.time() - start)
    print("WER: ", jiwer.wer(ground_truth, hypothesis))
 
    save_csv('result_transformer.csv', hypothesis)
    print('complete')
