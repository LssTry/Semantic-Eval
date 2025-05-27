import re
import json
import os
import sys
import time
import argparse
import numpy as np
import openai
import logging
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv

EMBED_MODEL = "text-embedding-3-large"

class TextSegmenter:
    def __init__(self, split_pattern=None, threshold=3):
        if split_pattern is None:
            self.split_pattern = r'(?<!\d)(?<=[.!?])\s+(?=["“‘\(A-Z])'
        else:
            self.split_pattern = split_pattern
        self.threshold = threshold

    def count_tokens(self, sample):
        return len(sample.split())

    def refine(self, raw_text):
        refined = re.sub(r'(\d+\.)\s+', r'\1 ', raw_text)
        segments = re.split(self.split_pattern, refined)
        final_segments = []
        for s in segments:
            s = s.strip()
            if re.match(r'^\d+\.$', s):
                continue
            if self.count_tokens(s) >= self.threshold:
                final_segments.append({"sentence": s})
        return final_segments

    def encode_json(self, data, label="sentences"):
        return json.dumps({label: data}, ensure_ascii=False, indent=4)

    def save(self, data, path, label="sentences"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({label: data}, f, ensure_ascii=False, indent=4)


class VectorSimilarity:
    def __init__(self, api_key=None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("Missing API key.")

    def embed(self, input_text, model=None):
        if not model:
            model = EMBED_MODEL
        try:
            r = openai.Embedding.create(input=input_text, model=model)
            return r['data'][0]['embedding']
        except openai.error.OpenAIError as e:
            logging.error(f"Error: {e}")
            return None

    def gather(self, block, prefix):
        found = []
        for i in block:
            for k, v in i.items():
                if k.startswith(prefix):
                    found.append(v)
                    break
        return found

    def normalize(self, vals):
        total = sum(vals)
        if total == 0:
            raise ValueError("Zero sum.")
        return np.array([v / total for v in vals])

    def package(self, records, name):
        p = self.gather(records, "sentence")
        storage = []
        for element in p:
            e = self.embed(element)
            if e:
                storage.append(e)
            else:
                logging.warning(f"Failed on: {element}")
        if storage:
            mat = np.array(storage)
        else:
            logging.warning(f"No embeddings found: {name}")
            mat = None
        return {"phrases": p, "matrix": mat}

    def compute_normalized_reweights(self, max_vals, ma, se):
        selected_weights_np = np.array(max_vals)
        if len(selected_weights_np) == 0:
            return []
        if len(selected_weights_np) == 1:
            return [1.0]
        ma_index = np.argmax(selected_weights_np)
        temp_weights = selected_weights_np.copy()
        temp_weights[ma_index] = -np.inf
        se_index = np.argmax(temp_weights)
        ma = ma
        se = se
        re = 1 - ma - se
        remaining_values = np.delete(selected_weights_np, [ma_index, se_index])
        num_remaining = len(remaining_values)
        if num_remaining == 0:
            return [ma, se]
        if np.sum(remaining_values) == 0:
            remaining = [re / num_remaining] * num_remaining
        else:
            total = np.sum(remaining_values)
            relative = remaining_values / total
            remaining = relative * re
        norm_w = remaining.tolist()
        norm_w.insert(se_index, se)
        norm_w.insert(ma_index, ma)
        return norm_w

    def finalize(self, c_data, l_data, w):
        c_mat = c_data.get("matrix")
        l_mat = l_data.get("matrix")
        if c_mat is None or l_mat is None:
            logging.error("Missing embeddings.")
            return None

        try:
            smat = cosine_similarity(c_mat, l_mat)
            smat = np.nan_to_num(smat)
        except Exception as e:
            logging.error(f"Similarity error: {e}")
            return None

        max_vals = smat.max(axis=1)
        index_vals = smat.argmax(axis=1)
        row_ids = np.arange(w.shape[0])
        sel_w = w[row_ids, index_vals]
        norm_w = self.compute_normalized_reweights(max_vals, ma=0.5, se=0.3)
        final_val = np.sum(norm_w * max_vals)
        final_val = round(final_val, 2)
        print(f"Score: {final_val}")
        return final_val

    def merge_weights(self, cw, lw):
        return (cw[:, np.newaxis] + lw) / 2


class GraphRanker:
    def __init__(self, threshold=0.1, alpha=0.85):
        self.threshold = threshold
        self.alpha = alpha

    def compute_smat(self, emb):
        return cosine_similarity(emb)

    def create_graph(self, sm):
        g = nx.Graph()
        n = sm.shape[0]
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if sm[i][j] >= self.threshold:
                    g.add_edge(i, j, weight=sm[i][j])
        return g

    def rank(self, g):
        return nx.pagerank(g, alpha=self.alpha, weight='weight')

    def scale(self, w):
        arr = np.array(list(w.values()))
        tot = arr.sum()
        if tot == 0:
            raise ValueError("Zero sum.")
        return arr / tot

    def process(self, emb):
        sm = self.compute_smat(emb)
        g = self.create_graph(sm)
        r = self.rank(g)
        return self.scale(r)


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--progress_file", type=str, default="progress.txt")
    p.add_argument("--excel_path", type=str, default="datasets/examples.xlsx")
    p.add_argument("--result_file", type=str, default="examples_step1.csv")
    p.add_argument("--api_key", type=str,
                   default="your api key")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--sleep_time", type=float, default=0.5)
    args = p.parse_args()
    prog_file = args.progress_file
    if os.path.exists(prog_file):
        with open(prog_file, 'r', encoding='utf-8') as f:
            z = f.read().strip()
            start_idx = int(z) if z else 0
    else:
        start_idx = 0
    x_path = args.excel_path
    if not os.path.exists(x_path):
        print(f"Not found: {x_path}")
        return
    data_frame = pd.read_excel(x_path)
    needed = {'candidate', 'reference'}
    if not needed.issubset(data_frame.columns):
        print("Missing columns.")
        return
    r_file = args.result_file
    h = not os.path.exists(r_file)
    vsim = VectorSimilarity(api_key=args.api_key)
    seg = TextSegmenter()
    gproc = GraphRanker()
    bsz = args.batch_size
    stime = args.sleep_time
    for start in range(start_idx, len(data_frame), bsz):
        finish = min(start + bsz, len(data_frame))
        batch_df = data_frame.iloc[start:finish]
        print(f"Processing from row {start} to {finish - 1}")
        for idx, row in batch_df.iterrows():
            try:
                lab_txt = str(row['candidate'])
                cand_txt = str(row['reference'])
                lab_sents = seg.refine(lab_txt)
                cand_sents = seg.refine(cand_txt)
                if not lab_sents or not cand_sents:
                    continue
                labVectorData = vsim.package(lab_sents, "Lab")
                candidateVectorData = vsim.package(cand_sents, "candidate")
                if labVectorData is None or candidateVectorData is None:
                    continue
                if labVectorData["matrix"] is None or candidateVectorData["matrix"] is None:
                    continue
                lw = gproc.process(labVectorData["matrix"])
                cw = gproc.process(candidateVectorData["matrix"])
                wmat = vsim.merge_weights(cw, lw)
                result = vsim.finalize(candidateVectorData, labVectorData, wmat)
                if result is None:
                    print(f"Row {idx} error.")
                    continue
                with open(r_file, 'a', newline='', encoding='utf-8') as of:
                    w = csv.writer(of)
                    if h:
                        w.writerow(["row_index", "reweight"])
                        h = False
                    w.writerow([idx, result])
                print(f"Row {idx} done.")
            except Exception as e:
                logging.error(f"Error at row {idx}: {e}")
                continue
        with open(prog_file, 'w', encoding='utf-8') as pf:
            pf.write(str(finish))
        print(f"Batch {start} to {finish - 1} finished.")
        print(f"Sleeping for {stime}s...")
        time.sleep(stime)


if __name__ == '__main__':
    run()
