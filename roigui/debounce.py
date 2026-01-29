"""Timer-based debouncing for expensive computations."""

from typing import Callable, Optional
from PySide6.QtCore import QObject, QTimer


class DebouncedComputation(QObject):
    """Base class for debounced computations.

    Subclasses implement compute() which reads current state and updates it.
    Call request() on each triggering event; computation fires after delay_ms
    of inactivity.

    Usage:
        class TraceComputer(DebouncedComputation):
            def compute(self):
                trace = expensive_trace_computation(self.state.roi)
                self.state.set_trace(trace)

        computer = TraceComputer(state, delay_ms=200)
        state.roi_changed.connect(computer.request)
    """

    def __init__(self, state, delay_ms: int = 200, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = state
        self.delay_ms = delay_ms

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    def request(self):
        """Request computation. Restarts timer on each call."""
        self._timer.start(self.delay_ms)

    def cancel(self):
        """Cancel pending computation."""
        self._timer.stop()

    def _on_timeout(self):
        """Timer fired - run computation."""
        self.compute()

    def compute(self):
        """Override in subclass to perform the actual computation."""
        raise NotImplementedError


class CallbackDebouncer(QObject):
    """Simple debouncer that calls a callback function.

    Usage:
        debouncer = CallbackDebouncer(my_expensive_function, delay_ms=200)
        some_signal.connect(debouncer.request)
    """

    def __init__(self, callback: Callable, delay_ms: int = 200,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self.callback = callback
        self.delay_ms = delay_ms

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    def request(self):
        """Request callback. Restarts timer on each call."""
        self._timer.start(self.delay_ms)

    def cancel(self):
        """Cancel pending callback."""
        self._timer.stop()

    def _on_timeout(self):
        self.callback()
