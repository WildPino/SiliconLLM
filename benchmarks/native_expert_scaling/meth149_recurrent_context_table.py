#!/usr/bin/env python3
"""Freeze recurrent causal token tuples before the METH-149 route replay."""

import argparse
from collections import Counter
import json
from pathlib import Path

import meth136_sparse_experts  # inserts S1 import path
import meth136_matched_sparse_train as M136


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    M136.bind_inputs()  # verifies every donor, checkpoint and training input hash
    _, chat, draws = M136.training_data()
    assert len(chat) == len(draws) == 256
    assert len({d["chat_index"] for d in draws}) == 256
    counter = Counter()
    positions = 0
    for _, ids, _ in chat:
        for position, token in enumerate(ids[:-1]):
            previous = ids[position - 1] if position else 0
            counter[(int(token), int(previous), position)] += 1
            positions += 1
    threshold = 32
    recurrent = sorted((token, previous, position, count)
                       for (token, previous, position), count in counter.items()
                       if count >= threshold)
    assert recurrent and all(count <= 256 for *_, count in recurrent)
    result = {"experiment": "METH-149-recurrent-context-table",
              "threshold_sequences": threshold,
              "chat_sequences": len(chat), "draws": len(draws),
              "chat_input_positions": positions,
              "recurrent_tuple_count": len(recurrent),
              "recurrent_events": sum(row[3] for row in recurrent),
              "tuples": [{"token": token, "previous": previous,
                          "position": position, "count": count}
                         for token, previous, position, count in recurrent],
              "source_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "raw_data_sha256": M136.M15.TRAIN_FILE_SHA,
              "train_chat_sha256": M136.M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": M136.M107.LONG_TEACHER_SHA,
              "draw_seed": M136.SEED}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": M136.digest(args.out),
                      "tuple_count": len(recurrent),
                      "recurrent_events": result["recurrent_events"],
                      "chat_input_positions": positions}), flush=True)


if __name__ == "__main__":
    main()
