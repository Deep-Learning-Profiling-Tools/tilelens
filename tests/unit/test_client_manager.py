import re
from contextlib import contextmanager

import pytest

from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client, ClientManager
from tilelens.core.config import config as cfg


class _DummyClient(Client):
    """Dummy client whose `self.tensors` comes from user input."""

    NAME = "dummy"

    def __init__(self, tensors=None, records=None):
        super().__init__()
        self.tensors = [] if tensors is None else list(tensors)
        self._records = [] if records is None else list(records)

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def arg_callback(self, name, arg, arg_cvt):
        return None

    def grid_callback(self, grid):
        return None

    def grid_idx_callback(self, grid_idx):
        return None

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return list(self._records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        return None


def test_client_manager_finalize_collects_client_tensors():
    """Test that the client manager collects any internal tensors that a client stores tensors.
    Important for the tracer + visualizer as the tracer stores tensors allocated in kernel scope,
    which the visualizer then displays.
    """
    tensor = object()
    record = object()
    manager = ClientManager([_DummyClient(tensors=[tensor], records=[record])])

    manager.finalize()

    assert tensor in manager.launch.tensors
    assert manager.launch.records == [record]


def test_add_clients_never_drops_or_replaces_a_client_silently():
    first = _DummyClient()
    manager = ClientManager([first])
    manager.add_clients([first])  # the same object again is a no-op

    class _SameName(_DummyClient):
        pass

    class _Other(_DummyClient):
        NAME = "other"

    for refused, named in (
        ([_DummyClient()], "'dummy' (_DummyClient), so another one (_DummyClient)"),
        ([_SameName()], "'dummy' (_DummyClient), so another one (_SameName)"),
        ([_Other(), _Other()], "'other' (_Other), so another one (_Other)"),
    ):
        with pytest.raises(ValueError, match=re.escape(named)):
            manager.add_clients(refused)
    # A refused batch adds nothing, its first client included.
    assert manager.clients == {"dummy": first}


def test_lock_fn_keeps_runtime_lock_decision():
    previous_num_sms = cfg.num_sms

    class LockCountingClient(_DummyClient):
        def __init__(self):
            super().__init__()
            self.lock_context_calls = 0

        @contextmanager
        def _lock_context(self):
            self.lock_context_calls += 1
            yield

    client = LockCountingClient()
    cfg.num_sms = 1
    wrapped = client.lock_fn(lambda: None)

    try:
        wrapped()
    finally:
        cfg.num_sms = previous_num_sms

    assert client.lock_context_calls == 1
