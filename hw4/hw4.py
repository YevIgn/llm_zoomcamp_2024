import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge import Rouge

github_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv"

model_name = "multi-qa-mpnet-base-dot-v1"


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    # Used to read data faster locally.
    csv_path = f"{github_url}?raw=1" if csv_path is None else csv_path
    df = pd.read_csv(csv_path)
    return df.iloc[:300]


def normalise(embedding: np.ndarray) -> np.ndarray:
    norm = np.sqrt((embedding * embedding).sum())
    result = embedding / norm
    return result


def avg_rouge_score(rouge_scores: dict) -> float:
    rouge_1 = rouge_scores["rouge-1"]["f"]
    rouge_2 = rouge_scores["rouge-2"]["f"]
    rouge_l = rouge_scores["rouge-l"]["f"]
    rouge_avg = (rouge_1 + rouge_2 + rouge_l) / 3
    return rouge_avg


if __name__ == "__main__":
    embedding_model = SentenceTransformer(model_name)
    df = load_dataset("hw4_q1.csv")

    # q1
    answer_llm = df.iloc[0].answer_llm
    q1_embedding = embedding_model.encode(answer_llm)
    print(q1_embedding[0])

    # Prepare Q2-Q3.
    answers_orig_embeddings = []
    answers_llm_embeddings = []
    answers_orig_embeddings_normalised = []
    answers_llm_embeddings_normalised = []
    scores = []
    scores_normalised = []
    for row in df.itertuples():
        answer_orig_embedding = embedding_model.encode(row.answer_orig)
        answer_llm_embedding = embedding_model.encode(row.answer_llm)
        answers_orig_embeddings.append(answer_orig_embedding)
        answers_llm_embeddings.append(answer_llm_embedding)
        answers_orig_embeddings_normalised.append(normalise(answer_orig_embedding))
        answers_llm_embeddings_normalised.append(normalise(answer_llm_embedding))
        scores.append(answer_llm_embedding.dot(answer_orig_embedding))
        scores_normalised.append(answers_llm_embeddings_normalised[-1].dot(answers_orig_embeddings_normalised[-1]))

    # q2
    print(np.percentile(scores, 75))

    # q3
    print(np.percentile(scores_normalised, 75))

    rouge_scorer = Rouge()
    element = df.iloc[10]
    q4_q5_rouge_scores = rouge_scorer.get_scores(element.answer_llm, element.answer_orig)[0]

    # q4
    print(q4_q5_rouge_scores["rouge-1"]["f"])  # Q4 answer.

    # q5
    print(avg_rouge_score(q4_q5_rouge_scores))

    rouge_scores_dicts = []
    for row in df.itertuples():
        rouge_scores = rouge_scorer.get_scores(row.answer_llm, row.answer_orig)[0]
        rouge_scores_dicts.append(
            {
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"],
            }
        )
    rouge_scores_df = pd.DataFrame(rouge_scores_dicts)

    # q6
    print(rouge_scores_df["rouge-2"].mean())
