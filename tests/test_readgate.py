import threading
import time

import pytest

from app.readgate import ReadGate


class TestCap:
    def test_never_exceeds_limit(self):
        gate = ReadGate(2)
        active = 0
        peak = 0
        lock = threading.Lock()

        def worker():
            nonlocal active, peak
            for _ in range(20):
                with gate.slot("diagnose"):
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    time.sleep(0.001)
                    with lock:
                        active -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak == 2  # cap fully used, never exceeded

    def test_invalid_size_rejected(self):
        with pytest.raises(ValueError):
            ReadGate(0)


class TestReshardPriority:
    def test_freed_slot_goes_to_waiting_reshard_first(self):
        gate = ReadGate(1)
        order = []

        def reshard():
            with gate.slot("reshard"):
                order.append("reshard")

        def diagnose():
            with gate.slot("diagnose"):
                order.append("diagnose")

        with gate.slot("diagnose"):  # main thread holds the only slot
            t1 = threading.Thread(target=reshard)
            t2 = threading.Thread(target=diagnose)
            t1.start()
            while gate.waiting_reshards < 1:
                time.sleep(0.001)
            t2.start()
            while gate.waiting_diagnoses < 1:
                time.sleep(0.001)
            # Both are queued now; releasing our slot must wake the reshard
            # reader first even though the diagnose reader is also waiting.
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive() and not t2.is_alive()
        assert order == ["reshard", "diagnose"]
        assert gate.available_slots == 1

    def test_diagnose_proceeds_when_no_reshard_waits(self):
        gate = ReadGate(1)
        with gate.slot("diagnose"):
            assert gate.available_slots == 0
        assert gate.available_slots == 1


class TestReleaseOnFailure:
    @pytest.mark.parametrize("kind", ["reshard", "diagnose"])
    def test_slot_released_when_read_raises(self, kind):
        gate = ReadGate(1)
        with pytest.raises(OSError, match="disk gone"):
            with gate.slot(kind):
                raise OSError("disk gone")
        assert gate.available_slots == 1
        # Gate still fully usable afterwards — nothing leaked.
        with gate.slot(kind):
            pass
