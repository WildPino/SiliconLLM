"""Only operational guards are inherited; target arithmetic lives elsewhere."""
import json
import meth490_r1_operations as operations

ROOT, DOC, write, stamp = operations.ROOT, operations.DOC, operations.write, operations.stamp
operations.BIND = DOC / 'meth493_binding.json'

class Context(operations.Context):
    def __init__(self, out, raw, seconds, bytes_limit):
        super().__init__(out, raw, seconds, bytes_limit)
        self.r['experiment'] = 'METH493 complete bank11 selected weighted-function target compilation'
    def deadline(self):
        self.abort(RuntimeError('METH493 hard wall deadline'))
    def guard(self):
        super().guard()
        total = 0
        for name in ('meth493_weighted_targets', 'meth493_retention'):
            folder = ROOT / 'results/native_expert_scaling' / name
            if folder.exists():
                total += sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        for p in DOC.glob('meth493*.json'):
            total += p.stat().st_size
        for p in (DOC/'RETENTION_493_20261006.json', DOC/'RETENTION_493_20261006.failure.json',
                  DOC/'ADMISSION_493_20261006.json'):
            if p.exists(): total += p.stat().st_size
        for p in (ROOT/'results/native_expert_scaling').glob('meth493*windows_terminal.json'):
            total += p.stat().st_size
        assert total <= 128 << 20, 'new_outputs128MiB'
    def finish(self, value):
        self.phase = 'terminal'; self.log(terminal=True); self.guard()
        for rel, sha in self.binding['preserved'].items(): assert self.digest(ROOT/rel) == sha
        assert self.digest(ROOT/'benchmarks/phase60/engine.c') == self.binding['engine_sha256']
        inventory = [{'path':str(p), 'bytes':p.stat().st_size, 'sha256':self.digest(p)}
                     for p in sorted(self.out.iterdir()) if p.is_file()]
        result = {**self.r, **value, 'end_utc':stamp(), 'resource':self.resources(),
                  'output_inventory':inventory, 'terminal_resource_path':str(self.out/'terminal_resources.json')}
        payload = (json.dumps(result, indent=2, allow_nan=False)+'\n').replace('\n','\r\n').encode()
        assert len(payload) <= 4 << 20; self.guard()
        with self.raw.open('xb') as stream: stream.write(payload)
        self.guard()
        terminal = {'raw_path':str(self.raw), 'raw_sha256':self.digest(self.raw),
                    'resource':self.resources(), 'scope':'After full terminal record write, before small receipt serialization.'}
        write(self.out/'terminal_resources.json', terminal); self.guard()
        self.done.set(); self.watchdog.cancel(); self.watcher.join(timeout=1)
        self.progress.close(); operations.faulthandler.disable(); self.fatal.close()
        self.terminal_resources = terminal
        return result
