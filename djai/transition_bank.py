from copy import deepcopy
import itertools
import chromadb
from chromadb.config import Settings
import networkx as nx
from matplotlib import pyplot as plt
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import pickle
from streamlit_agraph import agraph, Node, Edge, Config

class TransitionRetriever:

    def __init__(self, persist_directory: Path) -> None:
        chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(persist_directory.resolve())
            )
        )
        self.source_collection = chroma_client.get_collection(name="source_collection")
        self.dest_collection = chroma_client.get_collection(name="dest_collection")

    def get_chunk_data(self, name: str) -> dict:
        chunk_data = self.source_collection.get(
            where={"name": name},
            include=["documents", "metadatas", "embeddings"]
        )
        return chunk_data

    def get_candidates(self, name: str, n_candidates: int = 3) -> List[dict]:

        chunk_data = self.get_chunk_data(name) 

        candidate_list = []

        for chunk_idx in range(len(chunk_data["ids"])):

            chunk_embedding = chunk_data["embeddings"][chunk_idx]

            matches = self.dest_collection.query(
                query_embeddings=chunk_embedding,
                where={"name": {"$ne": name}},
                include=["metadatas", "distances"],
                n_results=n_candidates
            )

            for match_idx in range(len(matches["ids"])):
                candidate_list.append(
                    {
                        "dest_name": matches["metadatas"][0][match_idx]["name"],
                        "dest_from_time": matches["metadatas"][0][match_idx]["from_time"],
                        "dest_to_time": matches["metadatas"][0][match_idx]["to_time"],
                        "distance": matches["distances"][0][match_idx],
                        "source_from_time": chunk_data["metadatas"][chunk_idx]["from_time"],
                        "source_to_time": chunk_data["metadatas"][chunk_idx]["to_time"],
                        "source_name": chunk_data["metadatas"][chunk_idx]["name"]
                    }
                )    
        return candidate_list

    def get_songs(self) -> List[Path]:
        """
        Return all song names in the source collection
        """
        return list((Path(__file__).parents[1] / "processed_data").glob("*"))

class TransitionNetwork(nx.DiGraph):

    def __init__(self, retriever: Optional[TransitionRetriever] = None) -> None:
        super().__init__()
        self.retriever = retriever

    def _format_time(self, time: float) -> str:
        return f"{int(time // 60)}:{int(time % 60)}"

    def build(self) -> None:
        if self.retriever is None:
            raise ValueError("Retriever must be set before building the network")
        songs = self.retriever.get_songs()
        for song in tqdm(songs, desc="Building transition network"):
            song_name = song.name
            candidates = self.retriever.get_candidates(song_name)
            for candidate in candidates:
                source_time = f"{self._format_time(candidate['source_from_time'])}-{self._format_time(candidate['source_to_time'])}"
                dest_time = f"{self._format_time(candidate['dest_from_time'])}-{self._format_time(candidate['dest_to_time'])}"
                if self.has_edge(candidate['source_name'], candidate['dest_name']):
                    if self.get_edge_data(candidate['source_name'], candidate['dest_name'])['distance'] > candidate['distance']:
                        self[candidate['source_name']][candidate['dest_name']].update(
                            weight=candidate['distance']*100,
                            distance=candidate['distance'],
                            source_from_time=source_time,
                            dest_from_time=dest_time
                        )
                else:
                    self.add_edge(
                        candidate['source_name'],
                        candidate['dest_name'],
                        weight=candidate['distance']*100,
                        distance=candidate['distance'],
                        source_from_time=source_time,
                        dest_from_time=dest_time
                    )
        self.remove_nodes_from(list(nx.isolates(self)))


    def threshold(self, distance: float) -> 'TransitionNetwork':
        new_network = self.copy()
        edges_to_remove = [(u, v) for u, v, data in self.edges(data=True) if data['distance'] > distance]
        new_network.remove_edges_from(edges_to_remove)
        new_network.remove_nodes_from(list(nx.isolates(new_network)))
        return new_network

    def save(self, file_path: str) -> None:
        self.retriever.source_collection = None
        self.retriever.dest_collection = None
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path: str) -> 'TransitionNetwork':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_connected_nodes(self, node: str) -> List[str]:
        return list(nx.shortest_path(self, node).keys())

    def query(
        self, 
        source_node: str, 
        dest_node: str, 
        n_paths: int = 3,
        weight: str = None
        ) -> nx.DiGraph:
        """
        Given a source node and a destination node, return the subgraph of the combination of n shortest paths between them
        """

        if not self.has_node(source_node) or not self.has_node(dest_node):
            raise ValueError(f"Both nodes must exist in the graph. Got source_node: {source_node}, dest_node: {dest_node}")

        # Get n shortest paths
        paths = list(itertools.islice(nx.shortest_simple_paths(self, source_node, dest_node, weight=weight), n_paths))

        # Create new directed graph
        result_graph = TransitionNetwork()

        for path in paths:
            for i in range(len(path) - 1):
                # Get data associated with the edge
                data = self.get_edge_data(path[i], path[i+1])
                # Add edge to new graph
                result_graph.add_edge(path[i], path[i+1], **data)

        return result_graph


    def get_songs(self) -> List[str]:
        return list(self.nodes())

    def to_agraph(self):
        nodes = []
        edges = []
        for node in self.nodes:
            nodes.append(Node(id=node, label=node))
        for edge in self.edges:
            from_time = self.edges[edge]['source_from_time']
            to_time = self.edges[edge]['dest_from_time']
            edge_message = f"f: {from_time}\nt: {to_time}"
            edges.append(Edge(source=edge[0], target=edge[1], label=edge_message))
        config = Config(
            width=1000,
            height=1000,
            directed=True, 
            physics=False, 
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True,
            node={'labelProperty': 'label'},
            edge={'labelProperty': 'label'},
            tooltipDelay=100,
            )    
        return agraph(nodes, edges, config=config)
