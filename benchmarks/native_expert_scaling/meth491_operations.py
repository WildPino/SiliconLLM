"""Reuse frozen operational guards only; no source/main/math/audit import."""
import meth490_r1_operations as operations

ROOT, DOC, write, stamp = operations.ROOT, operations.DOC, operations.write, operations.stamp
operations.BIND = DOC / 'meth491_binding.json'

class Context(operations.Context):
    def __init__(self, out, raw, seconds, bytes_limit):
        super().__init__(out, raw, seconds, bytes_limit)
        self.r['experiment'] = 'METH491 parametric affine mass-interval witnesses'
    def deadline(self):
        self.abort(RuntimeError('METH491 hard wall deadline'))
