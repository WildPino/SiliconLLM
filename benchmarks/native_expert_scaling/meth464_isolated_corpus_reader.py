"""Arrow-only child. Corpus transport, no source selection or model imports."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import psutil

INDEX = struct.Struct('<QQQ32s')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--controls', type=Path, required=True)
    args = ap.parse_args(); start = time.monotonic(); process = psutil.Process(); peak = 0; count = 0
    def guard():
        nonlocal peak
        info = process.memory_info(); peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
        assert time.monotonic() - start <= 180 and peak <= 4 << 30, 'child180s4GiB'
        assert sum(p.stat().st_size for p in args.out.iterdir()) <= 2 << 30, 'output2GiB'
    with (args.out / 'reader.progress.jsonl').open('x', encoding='utf-8') as progress:
        def checkpoint(stage, **values):
            progress.write(json.dumps({'pid': os.getpid(), 'stage': stage, 'rows': count, 'seconds': time.monotonic() - start, **values}) + '\n')
            progress.flush(); guard()
        checkpoint('before_Arrow_import')
        import pyarrow
        import pyarrow.parquet as pq
        assert pyarrow.__version__ == '24.0.0'
        controls = json.loads(args.controls.read_text(encoding='utf-8'))
        checkpoint('before_ParquetFile')
        parquet = pq.ParquetFile(args.corpus)
        assert parquet.metadata.num_rows == 1243
        rows = []; known = {}; expected_offset = 16 + 1243 * INDEX.size
        with (args.out / 'corpus_utf8.bin').open('xb') as stream:
            stream.write(struct.pack('<8sII', b'M464TX01', 1243, INDEX.size)); stream.write(bytes(1243 * INDEX.size))
            for group in range(parquet.metadata.num_row_groups):
                checkpoint('before_single_thread_row_group', group=group, expected_rows=parquet.metadata.row_group(group).num_rows)
                table = parquet.read_row_group(group, columns=['text'], use_threads=False)
                assert table.num_rows == parquet.metadata.row_group(group).num_rows
                for chunk in table.column('text').chunks:
                    for scalar in chunk:
                        text = scalar.as_py(); assert isinstance(text, str)
                        data = text.encode('utf-8'); h = hashlib.sha256(data).digest()
                        assert stream.tell() == expected_offset
                        rows.append((stream.tell(), len(data), len(text), h)); stream.write(data)
                        expected_offset += len(data)
                        if str(count) in controls:
                            assert h.hex() == controls[str(count)], ('known_original_book', count)
                            known[str(count)] = h.hex()
                        count += 1; guard()
                del table
                checkpoint('single_thread_row_group_complete', group=group)
            assert count == 1243 and known == controls
            stream.seek(16)
            for row in rows: stream.write(INDEX.pack(*row))
            stream.flush()
        checkpoint('all_rows_and_known_sources_complete')
        summary = {'rows': count, 'row_groups': parquet.metadata.num_row_groups, 'wire_index_bytes': INDEX.size,
                   'known_original_book_sha256': known, 'spool_bytes': expected_offset,
                   'pyarrow': pyarrow.__version__, 'reader_method': 'isolated_read_row_group_text_use_threadsFalse',
                   'seconds': time.monotonic() - start, 'OS_peak_bytes': peak, 'pid': os.getpid()}
        with (args.out / 'reader.summary.json').open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2); stream.write('\n')
        guard()


if __name__ == '__main__': main()
