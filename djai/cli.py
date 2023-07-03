from djai.bank_loader import AudioProcessor
from djai.transition_bank import TransitionNetwork, TransitionRetriever
import typer
from loguru import logger
from pathlib import Path

cli = typer.Typer()

@cli.command()
def preprocess(
    raw_music_dir: Path, 
    processed_dir: Path,
    sample_rate: int = 24_000,
):
    """
    Preprocess the data

    Args:
        wav_dir (Path): Path to the directory containing the raw wav/flac files
        processed_dir (Path): Path to the directory to store the processed data
        sample_rate (int, optional): Sample rate to use for the processed data. Defaults to 24_000.
    """
    raw_music_dir = Path(raw_music_dir)
    processed_dir = Path(processed_dir)
    logger.info(f"Preprocessing data from {raw_music_dir} to {processed_dir}...")
    from djai.loading import process_folder
    process_folder(raw_music_dir, processed_dir, sr=sample_rate)

@cli.command()
def vectorize(
    processed_dir: Path,
    vector_store: Path
):
    """
    Vectorize the data

    Args:
        processed_dir (Path): Path to the existing processed data
        vector_store (Path): Path to the vector store to be created
    """
    processed_dir = Path(processed_dir)
    vector_store = Path(vector_store)
    logger.info(f"Vectorizing processed data in {processed_dir}, persisting vector store to: {vector_store}.")
    processor = AudioProcessor(data_folder=processed_dir, persist_directory=vector_store)
    processor.process_audio_files()


@cli.command()
def build_network(
    vector_store: Path,
    graph_path: Path
):
    """
    Build and save the full transition network as a pickled NetworkX graph

    Args:
        vector_store (Path): Path to the vector store
        graph_path (Path): Path to save the graph to e.g. "graph.pkl"
    """
    vector_store = Path(vector_store)
    graph_path = Path(graph_path)
    logger.info(f"Building transition network from vector store {vector_store}, saving to {graph_path}")
    retriever = TransitionRetriever(vector_store)
    network = TransitionNetwork(retriever)
    network.build()
    network.save(graph_path)
    logger.info(f"Saved transition network to {graph_path}")

@cli.command()
def interface(
    graph_path: Path,
    raw_music_dir: Path,
):
    """
    Run the interface

    Args:
        graph_path (Path): Path to the pickled NetworkX graph
        raw_music_dir (Path): Path to the directory containing the raw wav/flac files
    """
    graph_path = Path(graph_path)
    raw_music_dir = Path(raw_music_dir)
    logger.info(f"Running interface with graph from {graph_path} and raw music from {raw_music_dir}")
    from subprocess import run
    logger.info(f"Running streamlit command: streamlit run djai/interface.py -- --graph_path={str(graph_path)} --raw_music_dir={str(raw_music_dir)}")
    run(["streamlit", "run", "djai/interface.py", "--", f"--graph_path={str(graph_path)}", f"--raw_music_dir={str(raw_music_dir)}"])


@cli.command()
def pipeline(
    raw_music_dir: Path, 
    processed_dir: Path,
    vector_store: Path,
    graph_path: Path,
    sample_rate: int = 24_000,
):
    """
    Run the full pipeline

    Args:
        raw_music_dir (Path): Path to the directory containing the raw wav/flac files
        processed_dir (Path): Path to the directory to store the processed data
        vector_store (Path): Path to the vector store to be created
        graph_path (Path): Path to save the graph to e.g. "graph.pkl"
        sample_rate (int, optional): Sample rate to use for the processed data. Defaults to 24_000.
    """
    raw_music_dir = Path(raw_music_dir)
    processed_dir = Path(processed_dir)
    vector_store = Path(vector_store)
    graph_path = Path(graph_path)
    logger.info(f"Running pipeline with raw music from {raw_music_dir}, processed data in {processed_dir}, vector store in {vector_store}, graph in {graph_path}")
    preprocess(raw_music_dir, processed_dir, sample_rate)
    vectorize(processed_dir, vector_store)
    build_network(vector_store, graph_path)
    interface(graph_path, raw_music_dir)
