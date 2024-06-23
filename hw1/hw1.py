import requests
import tiktoken
from elasticsearch import Elasticsearch

docs_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1"
docs_index = "docs"

docs_index_mappings = {
    "properties": {
        "text": {
            "type": "text",
        },
        "section": {
            "type": "text",
        },
        "question": {
            "type": "text",
        },
        "course": {
            "type": "keyword",
        },
    }
}


def get_docs() -> list[dict]:
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    documents = []
    for course in documents_raw:
        course_name = course["course"]
        for doc in course["documents"]:
            doc["course"] = course_name
            documents.append(doc)
    return documents


def print_courses() -> None:
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    for course in documents_raw:
        print(course["course"])


def create_or_replace_docs_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=docs_index):
        es.indices.delete(index=docs_index)
    es.indices.create(index=docs_index, mappings=docs_index_mappings)


def add_docs_to_index(es: Elasticsearch) -> None:
    docs = get_docs()
    for doc in docs:
        es.index(
            index=docs_index,
            document=doc,
        )


q3_q4_query = "How do I execute a command in a running docker container?"


def prepare(es: Elasticsearch) -> None:
    create_or_replace_docs_index(es)
    add_docs_to_index(es)


def q3(es: Elasticsearch) -> None:
    q3_result = es.search(
        index=docs_index,
        query={
            "multi_match": {
                "query": q3_q4_query,
                "fields": ["question^4", "text"],
                "type": "best_fields",  # Default
            }
        },
    )
    print(q3_result)


def q4(es: Elasticsearch) -> list[dict]:
    q4_course = "machine-learning-zoomcamp"
    q4_results = es.search(
        index=docs_index,
        size=3,
        # sort={"_score": {"order": "desc"}},
        query={
            "bool": {
                "must": [
                    {"term": {"course": q4_course}},
                    {
                        "multi_match": {
                            "query": q3_q4_query,
                            "fields": ["question^4", "text"],
                            "type": "best_fields",  # Default
                        }
                    },
                ]
            }
        },
    )
    print(q4_results)
    q4_matched_question = q4_results["hits"]["hits"][2]["_source"]["question"]
    print("Question: ", q3_q4_query, " result: ", q4_matched_question)
    return q4_results["hits"]["hits"]


# Keeping indentation to avoid confusion between "expected" answers.
def q5(documents: list[dict]) -> str:
    context_entries = []
    for document in documents:
        context_entry = f"""
Q: {document["_source"]["question"]}
A: {document["_source"]["text"]}
        """.strip()
        context_entries.append(context_entry)
    context = "\n\n".join(context_entries)
    prompt = f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {q3_q4_query}

CONTEXT:
{context}
    """.strip()
    print(len(prompt))
    return prompt


def q6(prompt: str) -> None:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt)
    print(len(tokens))


# ToDo: Use OpenAI
def question_answering() -> None:
    pass


def main() -> None:
    es = Elasticsearch(
        hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
        verify_certs=False,
        basic_auth=("elastic", "changeme"),
    )
    # Comment/uncomment below during re-execution.
    prepare(es)
    q3(es)
    documents = q4(es)
    prompt = q5(documents)
    q6(prompt)


if __name__ == "__main__":
    main()
