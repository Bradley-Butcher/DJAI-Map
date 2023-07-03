import chromadb
from pathlib import Path
from tqdm import tqdm
from djai.loading import get_processed_data, AudioChunk
from djai.model import MusicVectorizer
from chromadb.config import Settings
import shutil
from loguru import logger
from typer import prompt


class AudioProcessor:
    def __init__(
        self, 
        data_folder: Path,
        persist_directory: Path = Path(__file__).parents[1] / "vector_store"
    ):
        self.data_folder = data_folder
        self.vectoriser = MusicVectorizer()
        if persist_directory.exists():
            remove = prompt(f"Directory {persist_directory} already exists. Remove? (y/n)")
            if remove.lower() == "y":
                shutil.rmtree(persist_directory)
            else:
                raise ValueError(f"Directory {persist_directory} already exists. Remove it or choose another directory.")
        self.chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(persist_directory.resolve())
            )
        )

    def _vectorize_and_store(self, audio_chunk: AudioChunk, audio_track_name: str, collection):
        embeddings = self.vectoriser(audio_chunk.audio)
        collection.add(
            embeddings=embeddings,
            documents=audio_track_name,
            metadatas={"name": audio_track_name, "from_time": audio_chunk.from_time, "to_time": audio_chunk.to_time},
            ids=f"{audio_track_name}_{audio_chunk.from_time}_{audio_chunk.to_time}"
        )

    def process_audio_files(self):
        source_collection = self.chroma_client.create_collection(name="source_collection")
        dest_collection = self.chroma_client.create_collection(name="dest_collection")
        total_files = len(list(self.data_folder.glob("*")))

        with tqdm(total=total_files, desc="Processing audio files") as pbar:
            for audio_track in get_processed_data(self.data_folder):
                for source_chunk in audio_track.get_chunks(ctype="source"):
                    pbar.set_postfix({"current_file - source": audio_track.name}, refresh=True)
                    self._vectorize_and_store(source_chunk, audio_track.name, source_collection)
                for dest_chunk in audio_track.get_chunks(ctype="dest"):
                    pbar.set_postfix({"current_file - dest": audio_track.name}, refresh=True)
                    self._vectorize_and_store(dest_chunk, audio_track.name, dest_collection)
                pbar.update()
        logger.info(f"Vectorized and stored {total_files} audio files.")
        self.chroma_client.close()