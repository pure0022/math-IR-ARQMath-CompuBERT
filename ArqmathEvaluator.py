import numpy as np
from arqmath_eval import get_judged_documents
from arqmath_eval import get_topics
from arqmath_eval.common import get_ndcg
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import os
from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator
from typing import List


class ArqmathEvaluator(EmbeddingSimilarityEvaluator):
    """
    Lightweight implementation of IREvaluator - without content indexing,
    Utilizing the get_ndcg for small validation set of Task1-votes

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    index = None
    finalized_index = False
    trec_metric = "euclidean_ndcg"
    task = 'task1-votes'
    subset = 'small-validation'

    def __init__(self, model: SentenceTransformer, 
                 sentences1: List[str], sentences2: List[str], scores: List[float], 
                 post_parser_postproc: PostParserRecord,
                 batch_size: int = 8,
                 main_similarity: SimilarityFunction = None, 
                 name: str = '', 
                 show_progress_bar: bool = None, 
                 write_csv: bool = True):

        super().__init__(sentences1, sentences2, scores, 
                         batch_size=batch_size, 
                         main_similarity=main_similarity, 
                         name=name, 
                         show_progress_bar=show_progress_bar,
                         write_csv=write_csv)
        self.model = model
        self.batch_size=batch_size
        self.post_parser = post_parser_postproc

    @staticmethod
    def report_ndcg_results(result_tsv_name: str, results: dict):
        with open(result_tsv_name, 'wt') as f:
            for topic, documents in results.items():
                top_documents = sorted(documents.items(), key=lambda x: x[1], reverse=True)[:1000]
                for rank, (document, similarity_score) in enumerate(top_documents):
                    line = '{}\txxx\t{}\t{}\t{}\txxx'.format(topic, document, rank + 1, similarity_score)
                    # print(line, file=f)

    def eval_transformer(self, subsample: int = False):
        results = {}
        all_questions_ids = get_topics(task=self.task, subset=self.subset)
        all_questions = dict([(int(qid), self.post_parser.map_questions[int(qid)]) for qid in all_questions_ids])
        # all_questions = dict([(int(qid), "Question %s content" % qid) for qid in all_questions_ids])
        if subsample:
            all_questions = all_questions[:subsample]

        for i, (qid, question) in enumerate(all_questions.items()):
            results[str(qid)] = {}
            judged_answer_ids = get_judged_documents(task=self.task, subset=self.subset, topic=str(qid))
            question_e = self.model.encode([question.body], batch_size=self.batch_size)
            try:
                answers_bodies = [self.post_parser.map_just_answers[int(aid)].body for aid in judged_answer_ids]
                # answers_bodies = ["Answer %s body" % aid for aid in judged_answer_ids]
            except KeyError:
                print("Key error at qid %s" % qid)
                answers_bodies = []
                # answers_bodies = ["Answer %s body" % aid for aid in judged_answer_ids]
            if not answers_bodies:
                print("No evaluated answers for question %s, dtype %s" % (qid, str(type(qid))))
                continue
            answers_e = self.model.encode(answers_bodies, batch_size=self.batch_size)
            answers_dists = cosine_similarity(np.array(question_e), np.array(answers_e))[0]

            for aid, answer_sim in sorted(zip(judged_answer_ids, answers_dists), key=lambda qid_dist: qid_dist[1],
                                          reverse=True):
                print(aid, answer_sim)
                results[str(qid)][str(aid)] = float(answer_sim)

        return results

    def __call__(self, *args, eval_all_metrics=True, report_tsv: str = False, **kwargs) -> float:
        def trec_metric_f():
            results = self.eval_transformer()
            ndcg_val = get_ndcg(results, task=self.task, subset=self.subset)
            print("Computed ncdg: %s" % ndcg_val)
            if report_tsv:
                self.report_ndcg_results(report_tsv, results)
            return ndcg_val

        if eval_all_metrics:
            return super(ArqmathEvaluator, self).__call__(*args, **kwargs)
        else:
            return trec_metric_f()

