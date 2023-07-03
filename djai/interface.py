from pathlib import Path
import streamlit as st
import networkx as nx
from djai.transition_bank import TransitionNetwork
from loguru import logger
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--graph_path", type=str, default="graph.pkl")
parser.add_argument("--raw_music_dir", type=str, default="data")
try:
    args = parser.parse_args()
except SystemExit as e:
    os._exit(e.code)

trans_graph = TransitionNetwork.from_file(Path(args.graph_path))
raw_music_dir = Path(args.raw_music_dir)

with st.sidebar:
    st.title("DJ AI")
    st.subheader("Transition Bank")
    st.slider("Max L2 Distance", 0., 10., 3.5, key="threshold")
    source_song = st.selectbox("Select a source song", trans_graph.threshold(st.session_state.threshold).get_songs())
    dest_songs = []
    try:
        dest_songs = trans_graph.threshold(st.session_state.threshold).get_connected_nodes(source_song)
    except nx.NetworkXNoPath:
        logger.warning(f"No path found for {source_song}")
    dest_song = st.selectbox("Select a destination song", dest_songs)

if source_song is not None and dest_song is not None and source_song != dest_song:
    graph = trans_graph.threshold(st.session_state.threshold).query(source_song, dest_song).to_agraph()
    if graph is not None:
        st.audio(str(Path(__file__).parents[1] / "data" / f"{graph}.wav"))

        
    