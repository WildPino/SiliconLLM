"""Unchanged frozen numerical body; corrected compiler apparatus/new namespace."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth528_prefix_bound_repair2 as frozen
from meth528_prefix_io_repair3 import PrefixContext as RepairContext, build


def context(kind,manifest_sha,freeze):
    assert kind=='prefix_bound_repair2'
    return RepairContext('prefix_bound_repair3',manifest_sha,freeze)


frozen.PrefixContext=context
frozen.build=build
if __name__=='__main__':frozen.main()
