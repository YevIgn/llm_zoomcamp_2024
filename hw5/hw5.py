from elasticsearch import Elasticsearch

index_name = "documents_20240820_5039"


def search(query_text: str) -> None:
    es_client = Elasticsearch(
        hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}],
        verify_certs=False,
        basic_auth=("elastic", "changeme"),
    )
    response = es_client.search(
        index=index_name,
        size=5,
        query={
            "match": {
                "question": {
                    "query": query_text
                }
            }
        },
    )
    for hit in response['hits']['hits']:
        print(hit['_source'])


if __name__ == "__main__":
    query_text = "When is the next cohort?"
    search(query_text)
