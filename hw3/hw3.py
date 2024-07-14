from abc import abstractmethod, ABC
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

BASE_URL = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
INDEX_NAME = "course-questions"
COURSE = "machine-learning-zoomcamp"
INDEX_DIMS = 768
DEFAULT_NUM_RESULTS = 5
Q1_QUESTION = "I just discovered the course. Can I still join it?"

Document = dict
Relevance = list[bool]
SearchFunction = Callable[[str], list[dict]]


def get_model() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-distilbert-cos-v1")


def load_documents(base_url: str, course: str) -> list[Document]:
    relative_url = "03-vector-search/eval/documents-with-ids.json"
    docs_url = f"{base_url}/{relative_url}?raw=1"
    docs_response = requests.get(docs_url)
    documents = [document for document in docs_response.json() if document["course"] == course]
    return documents


def load_ground_truth(base_url: str, course: str) -> list[dict]:
    relative_url = "03-vector-search/eval/ground-truth-data.csv"
    ground_truth_url = f"{base_url}/{relative_url}?raw=1"
    df_ground_truth = pd.read_csv(ground_truth_url)
    df_ground_truth = df_ground_truth[df_ground_truth.course == course]
    ground_truth = df_ground_truth.to_dict(orient="records")
    return ground_truth


def create_index(es_client: Elasticsearch, index_name: str, index_dims: int) -> None:
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "id": {"type": "keyword"},
                "question_vector": {"type": "dense_vector", "dims": index_dims, "index": True, "similarity": "cosine"},
                "text_vector": {"type": "dense_vector", "dims": index_dims, "index": True, "similarity": "cosine"},
                "question_text_vector": {
                    "type": "dense_vector",
                    "dims": index_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)


def q1(model: SentenceTransformer) -> np.ndarray:
    embedding = model.encode(Q1_QUESTION)
    print(embedding)
    return embedding


def doc_to_qa_text(document: Document) -> str:
    return f'{document["question"]} {document["text"]}'


def q2(model: SentenceTransformer, documents: list[Document]) -> np.ndarray:
    embeddings = []
    for document in documents:
        qa_text = doc_to_qa_text(document)
        embedding = model.encode(qa_text)
        document["question_text_vector"] = embedding
        embeddings.append(embedding)
    X = np.array(embeddings)
    print(X.shape)
    return X


@dataclass
class VectorSearchEngine:
    documents: list[Document]
    embeddings: np.ndarray

    def search(self, v_query: np.ndarray, num_results: int = DEFAULT_NUM_RESULTS) -> list[Document]:
        scores = self.embeddings.dot(v_query)
        idx = np.argpartition(-scores, num_results)[:num_results]
        return [self.documents[i] for i in idx]


def hit_rate(relevance_total: list[Relevance]):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)


def mean_reciprocal_rank(relevance_total: list[Relevance]):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


def evaluate(ground_truth: list[dict], search_function: SearchFunction) -> dict:
    relevance_total = []
    for question in ground_truth:
        doc_id = question["document"]
        results = search_function(question["question"])
        relevance = [document["id"] == doc_id for document in results]
        relevance_total.append(relevance)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mean_reciprocal_rank(relevance_total),
    }


@dataclass
class QuestionSearchEngine:
    model: SentenceTransformer
    vector_search_engine: VectorSearchEngine

    def search(self, question: str, num_results: int = DEFAULT_NUM_RESULTS) -> list[Document]:
        v_query = self.model.encode(question)
        return self.vector_search_engine.search(v_query, num_results)


def q3(q1_v_query: np.ndarray, embeddings: np.ndarray) -> None:
    scores = embeddings.dot(q1_v_query)
    print(np.max(scores))


def q4(ground_truth: list[dict], search_engine: QuestionSearchEngine) -> None:
    print(evaluate(ground_truth=ground_truth, search_function=search_engine.search))


def populate_index(
    es_client: Elasticsearch,
    model: SentenceTransformer,
    index_name: str,
    documents: list[Document],
) -> None:
    for document in documents:
        question = document["question"]
        text = document["text"]
        document["question_vector"] = model.encode(question)
        document["text_vector"] = model.encode(text)
        es_client.index(index=index_name, document=document)


class ElasticSearchKnnQueryBuilder(ABC):
    @abstractmethod
    def build_search_query(self, query_vector: np.ndarray) -> dict:
        pass


@dataclass
class SimpleEsKnnQueryBuilder(ElasticSearchKnnQueryBuilder):
    field: str
    course: str
    num_results: int = DEFAULT_NUM_RESULTS

    def build_search_query(self, query_vector: np.ndarray) -> dict:
        knn = {
            "field": self.field,
            "query_vector": query_vector,
            "k": self.num_results,
            "num_candidates": 10000,
            "filter": {"term": {"course": self.course}},
        }
        return {"knn": knn, "_source": ["text", "section", "question", "course", "id"]}


@dataclass
class CombinedEsKnnQueryBuilder(ElasticSearchKnnQueryBuilder):
    course: str
    num_results: int = DEFAULT_NUM_RESULTS

    def build_search_query(self, query_vector: np.ndarray) -> dict:
        return {
            "size": self.num_results,
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"term": {"course": self.course}},
                                "script": {
                                    "source": """
                                    cosineSimilarity(params.query_vector, 'question_vector') + 
                                    cosineSimilarity(params.query_vector, 'text_vector') + 
                                    cosineSimilarity(params.query_vector, 'question_text_vector') + 
                                    1
                                """,
                                    "params": {"query_vector": query_vector},
                                },
                            }
                        }
                    ],
                    "filter": {"term": {"course": self.course}},
                }
            },
            "_source": ["text", "section", "question", "course", "id"],
        }


@dataclass
class EsSearchEngine:
    model: SentenceTransformer
    es_client: Elasticsearch
    knn_search_query_builder: ElasticSearchKnnQueryBuilder
    index_name: str = INDEX_NAME

    def search(self, question: str, verbose: bool = False) -> list[Document]:
        question_vector = self.model.encode(question)
        search_query = self.knn_search_query_builder.build_search_query(question_vector)
        es_results = self.es_client.search(index=self.index_name, body=search_query)
        if verbose:
            print(es_results)
        result_docs = [hit["_source"] for hit in es_results["hits"]["hits"]]
        return result_docs


def q5(es_search_engine: EsSearchEngine) -> None:
    es_search_engine.search(
        question=Q1_QUESTION,
        verbose=True,
    )


def q6(ground_truth: list[dict], search_engine: EsSearchEngine) -> None:
    print(evaluate(ground_truth=ground_truth, search_function=search_engine.search))


def main() -> None:
    es_client = Elasticsearch(
        hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
        verify_certs=False,
        basic_auth=("elastic", "changeme"),
    )
    create_index(
        es_client=es_client,
        index_dims=INDEX_DIMS,
        index_name=INDEX_NAME,
    )
    model = get_model()

    print("Q1")
    q1_v_query = q1(model)
    print("------")

    print("\nQ2")
    documents = load_documents(
        base_url=BASE_URL,
        course=COURSE,
    )
    embeddings = q2(model, documents)
    print("------")

    print("\nQ3")
    q3(q1_v_query=q1_v_query, embeddings=embeddings)
    print("------")

    print("\nQ4")
    vector_search_engine = VectorSearchEngine(documents=documents, embeddings=embeddings)
    ground_truth = load_ground_truth(
        base_url=BASE_URL,
        course=COURSE,
    )
    question_search_engine = QuestionSearchEngine(model, vector_search_engine)
    q4(ground_truth=ground_truth, search_engine=question_search_engine)
    print("------")

    print("\nQ5")
    populate_index(
        es_client=es_client,
        model=model,
        index_name=INDEX_NAME,
        documents=documents,
    )
    knn_search_query_builder = CombinedEsKnnQueryBuilder(course=COURSE)
    es_search_engine = EsSearchEngine(
        model=model,
        es_client=es_client,
        knn_search_query_builder=knn_search_query_builder,
    )
    q5(es_search_engine=es_search_engine)
    print("------")

    print("\nQ6")

    q6(ground_truth=ground_truth, search_engine=es_search_engine)
    print("------")


if __name__ == "__main__":
    main()
