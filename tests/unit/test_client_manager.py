from contextlib import contextmanager

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
        self.block_start_calls = 0

    def pre_run_callback(self, fn):
        return True

    def block_start_callback(self, fn):
        self.block_start_calls += 1

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


def test_client_manager_only_marks_block_started_after_all_clients_accept():
    class VetoClient(_DummyClient):
        NAME = "veto"

        def pre_run_callback(self, fn):
            return False

    accepted = _DummyClient()
    veto = VetoClient()
    manager = ClientManager([accepted, veto])

    assert manager.pre_run_callback(lambda: None) is False
    assert accepted.block_start_calls == 0
    assert veto.block_start_calls == 0

    manager = ClientManager([accepted])
    assert manager.pre_run_callback(lambda: None) is True
    assert accepted.block_start_calls == 1


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
