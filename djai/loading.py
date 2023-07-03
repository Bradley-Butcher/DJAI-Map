from pathlib import Path
import pickle
from typing import Generator
from typing_extensions import Literal
import numpy as np
import torch
import soundfile as sf
import librosa
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

ACCEPTED_EXTENSIONS = ["wav", "flac"]

@dataclass
class AudioChunk:
    audio: torch.Tensor
    from_time: float
    to_time: float

    def minute_second(self) -> str:
        return f"{int(self.from_time // 60)}:{int(self.from_time % 60)}-{int(self.to_time // 60)}:{int(self.to_time % 60)}"

def chunk(audio: np.array, chunk_size: float, sample_rate: int = 24000,
          overlap_ratio: float = 0., time_offset: float = 0.,
          pad_last: bool = False) -> Generator[AudioChunk, None, None]:
    
    chunk_size = int(chunk_size * sample_rate) 
    overlap = int(chunk_size * overlap_ratio)
    total_length = len(audio) 

    for start in range(0, total_length, chunk_size - overlap):
        end = start + chunk_size
        if end > total_length:
            chunk_ = np.pad(audio[start:], (0, end - total_length)) if pad_last else audio[start:]
        else:
            chunk_ = audio[start:end]

        yield AudioChunk(
            audio=torch.from_numpy(chunk_),
            from_time=time_offset + (start / sample_rate),
            to_time=time_offset + (end / sample_rate)
        )

@dataclass
class AudioFile:
    source: np.array
    dest: np.array
    name: str
    sr: int
    original_length: int
    source_offset: float
    dest_offset: float = 0.0

    @staticmethod
    def load_audio(filepath: Path, sr: int = 24000) -> tuple[np.array, int]:
        data, samplerate = sf.read(str(filepath), dtype='float32')
        if samplerate != sr:
            data = librosa.resample(data.T, orig_sr=samplerate, target_sr=sr, res_type='soxr_hq').T
        return data, sr

    @classmethod
    def load(cls, filepath: Path) -> 'AudioFile':
        source_file = filepath / "source.wav"
        dest_file = filepath / "dest.wav"
        obj_file = filepath / f'{filepath.stem}_audiofile.pkl'

        if not source_file.exists() or not dest_file.exists() or not obj_file.exists():
            raise ValueError(f"Both source, destination and object files must exist. Check your path: {filepath}")

        with open(obj_file, 'rb') as f:
            obj = pickle.load(f)

        obj.source, _ = cls.load_audio(source_file, obj.sr)
        obj.dest, _ = cls.load_audio(dest_file, obj.sr)
        return obj

    @classmethod
    def from_raw(cls, filepath: Path, sr: int = 24000, source_cutoff: float = 0.8,
                 dest_cutoff: float = 0.2) -> 'AudioFile':
        raw, _ = cls.load_audio(filepath, sr)
        original_length = len(raw)
        raw = raw[int(original_length * 0.05):int(original_length * 0.9)]
        total_length = len(raw)
        source = raw[int(total_length * source_cutoff):]
        dest = raw[:int(total_length * dest_cutoff)]

        return cls(
            source=source, 
            dest=dest, 
            name=filepath.stem, 
            sr=sr,
            original_length=original_length,
            source_offset=original_length * source_cutoff / sr
        )

    def save(self, save_folder: Path) -> None:
        save_folder = save_folder / self.name
        save_folder.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_folder / "source.wav"), self.source, samplerate=self.sr)
        sf.write(str(save_folder / "dest.wav"), self.dest, samplerate=self.sr)
        with open(save_folder / f'{self.name}_audiofile.pkl', 'wb') as f:
            pickle.dump(self, f)

    def get_chunks(self, ctype: Literal["source", "dest"] = "source", 
                   chunk_size: float = 5.0, overlap_ratio: float = 0.5, 
                   pad_last: bool = False) -> Generator[AudioChunk, None, None]:

        audio = self.source if ctype == "source" else self.dest
        time_offset = self.source_offset if ctype == "source" else self.dest_offset
        return chunk(audio=audio, chunk_size=chunk_size, sample_rate=self.sr,
                     overlap_ratio=overlap_ratio, pad_last=pad_last,
                     time_offset=time_offset)

def process_folder(
    folder_path: Path, 
    output_folder: Path, 
    sr: int = 24000
) -> None:
    files = []
    for ext in ACCEPTED_EXTENSIONS:
        files.extend(folder_path.glob(f"*.{ext}"))
    logger.info(f"Found {len(files)} files")
    output_folder.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(files), desc="Processing music files") as pbar:
        for filepath in files:
            audio_file = AudioFile.from_raw(filepath, sr)
            audio_file.save(output_folder)
            pbar.set_postfix({"current_file": filepath.name}, refresh=True)
            pbar.update()
    logger.info(f"Processed {len(files)} files")

def get_processed_data(folder_path: Path) -> Generator[AudioFile, None, None]:
    filepaths = [f for f in folder_path.glob("*") if f.is_dir()]
    for filepath in filepaths:
        yield AudioFile.load(filepath)