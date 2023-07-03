# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn

class MusicVectorizer:
    def __init__(self) -> None:
        self.model, self.processor = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name_or_path: str = "m-a-p/MERT-v1-330M"):
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        return model, processor

    def __call__(self, audio: torch.tensor, sample_rate: int = 24_000):
        mono_audio = audio.mean(dim=1)
        input = self.processor(mono_audio, return_tensors="pt", sampling_rate=sample_rate).to(self.device)
        with torch.no_grad():
            outputs = self.model(**input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        outputs = outputs.mean(dim=1)
        return outputs.squeeze(0).tolist()