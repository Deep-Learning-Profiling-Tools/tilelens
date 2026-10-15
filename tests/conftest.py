import pytest


def pytest_addoption(parser):
    group = parser.getgroup("tilelens")
    group.addoption(
        "--triton-kernels-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Device mode for tests/end_to_end/test_triton_kernels.py. "
            "'cpu' runs CPU fake-tensor checks for CI without CUDA."
        ),
    )


@pytest.fixture
def unreachable_driver(monkeypatch):
    """``unreachable_driver(message)`` makes Triton's active driver
    unreachable, as on a machine without a GPU: any question to it raises
    ``AssertionError(message)``. One call reaches the real driver: unloading
    a module an earlier test loaded on a real GPU, which Triton 3.8's
    CompiledKernel.__del__ does through the driver whenever that kernel is
    collected (e.g. when tilelens.clear() drops the launch holding it)."""
    from triton.runtime.driver import driver

    owner = type(driver)
    real = owner.__dict__["active"]

    def refuse(message: str) -> None:
        class Utils:
            def unload_module(self, module):
                return real.__get__(driver, owner).utils.unload_module(module)

            def __getattr__(self, name):
                raise AssertionError(message)

        class Active:
            utils = Utils()

            def __getattr__(self, name):
                raise AssertionError(message)

        stand_in = Active()
        monkeypatch.setattr(owner, "active", property(lambda self: stand_in))

    return refuse


@pytest.fixture(scope="session", params=["cpu"])
def device(request):
    return request.param
