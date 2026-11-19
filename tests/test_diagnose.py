import contextlib
import json
import math
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.engine import execute_job
from app.jobs import Job, JobStore, RankState, compute_job_id
from app.main import Settings, create_app
from app.manifest import load_manifest
from app.plans import rank_intervals
from app.readgate import ReadGate
from tests.conftest import floats_to_bytes

TOTAL = 1000
STEP_OLD = 10
STEP_NEW = 11


def write_custom(root, total, step, w_bounds, m_bounds, w_vals, m_vals):
    """Like tests.conftest.write_source but with explicit per-element values."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"step": step, "total_length": total}
    for kind, bounds, vals in (("weights", w_bounds, w_vals), ("momentum", m_bounds, m_vals)):
        assert bounds[0] == 0 and bounds[-1] == total
        (root / kind).mkdir(exist_ok=True)
        segments = []
        for idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
            path = f"{kind}/part-{idx}.f32"
            (root / path).write_bytes(floats_to_bytes(vals[start:end]))
            segments.append({"path": path, "start": start, "end": end})
        manifest[kind] = segments
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def make_pair(tmp_path, total=TOTAL, mutate=None):
    """Old/new checkpoint pair; all four shard layouts differ on purpose."""
    w_old = [float(i) for i in range(total)]
    m_old = [-0.5 * i for i in range(total)]
    w_new, m_new = list(w_old), list(m_old)
    if mutate:
        mutate(w_old, m_old, w_new, m_new)
    old = tmp_path / "old"
    new = tmp_path / "new"
    write_custom(old, total, STEP_OLD, [0, 400, total], [0, 250, 800, total], w_old, m_old)
    write_custom(new, total, STEP_NEW, [0, 333, 777, total], [0, 500, total], w_new, m_new)
    return old, new


def make_client(tmp_path, old, new, chunk_bytes=64, sync=True, gate=None, read_concurrency=4):
    settings = Settings(
        source_dir=old,
        next_dir=new,
        output_root=tmp_path / "out",
        chunk_bytes=chunk_bytes,
        read_concurrency=read_concurrency,
        sync_execution=sync,
    )
    app = create_app(settings, gate=gate)
    return TestClient(app)


class TestClassification:
    def test_three_categories_across_mismatched_shards(self, tmp_path):
        def mutate(w_old, m_old, w_new, m_new):
            w_new[64] = math.nan  # weights only, on a 16-element window boundary
            w_new[777] = math.inf  # weights only, on a NEW weights shard boundary
            m_new[400] = math.nan  # momentum only, on an OLD weights shard boundary
            m_new[999] = -math.inf  # momentum only, last element (partial tail window)
            w_new[500] = math.nan  # both
            m_new[500] = math.inf

        old, new = make_pair(tmp_path, mutate=mutate)
        with make_client(tmp_path, old, new) as client:
            resp = client.post("/diagnose", json={})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["step_old"] == STEP_OLD
            assert body["step_new"] == STEP_NEW
            assert body["total_length"] == TOTAL
            assert body["weights_only"] == {"count": 2, "first_indices": [64, 777]}
            assert body["momentum_only"] == {"count": 2, "first_indices": [400, 999]}
            assert body["both"] == {"count": 1, "first_indices": [500]}

    def test_explicit_dirs_override_defaults(self, tmp_path):
        old, new = make_pair(tmp_path)
        # Settings point at nonexistent dirs; explicit body must win.
        settings = Settings(
            source_dir=tmp_path / "nope",
            next_dir=tmp_path / "nope2",
            output_root=tmp_path / "out",
            chunk_bytes=128,
            sync_execution=True,
        )
        with TestClient(create_app(settings)) as client:
            resp = client.post("/diagnose", json={"old_dir": str(old), "new_dir": str(new)})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["weights_only"]["count"] == 0
            assert body["momentum_only"]["count"] == 0
            assert body["both"]["count"] == 0

    def test_first_indices_capped_at_five_smallest(self, tmp_path):
        spots = [3, 40, 120, 300, 511, 700, 850, 999]

        def mutate(w_old, m_old, w_new, m_new):
            for i in spots:
                w_new[i] = math.nan

        old, new = make_pair(tmp_path, mutate=mutate)
        with make_client(tmp_path, old, new) as client:
            body = client.post("/diagnose", json={}).json()
            assert body["weights_only"]["count"] == len(spots)
            assert body["weights_only"]["first_indices"] == spots[:5]

    def test_persistent_or_recovered_anomalies_not_counted(self, tmp_path):
        def mutate(w_old, m_old, w_new, m_new):
            w_old[7] = math.nan
            w_new[7] = math.nan  # already broken before: not new
            m_old[8] = math.inf
            m_new[8] = 1.0  # recovered: not an anomaly
            m_old[500] = math.nan
            m_new[500] = math.nan  # persistent in momentum too
            w_new[9] = math.nan  # genuinely new

        old, new = make_pair(tmp_path, mutate=mutate)
        with make_client(tmp_path, old, new) as client:
            body = client.post("/diagnose", json={}).json()
            assert body["weights_only"] == {"count": 1, "first_indices": [9]}
            assert body["momentum_only"] == {"count": 0, "first_indices": []}
            assert body["both"] == {"count": 0, "first_indices": []}


class TestValidation:
    def test_length_mismatch_rejected_422(self, tmp_path):
        old, _ = make_pair(tmp_path)
        other = tmp_path / "other"
        write_custom(
            other, TOTAL + 1, STEP_NEW, [0, TOTAL + 1], [0, TOTAL + 1],
            [1.0] * (TOTAL + 1), [1.0] * (TOTAL + 1),
        )
        with make_client(tmp_path, old, other) as client:
            resp = client.post("/diagnose", json={})
            assert resp.status_code == 422
            assert "differ in length" in resp.json()["detail"]

    def test_invalid_manifest_rejected_422(self, tmp_path):
        old, new = make_pair(tmp_path)
        manifest = json.loads((new / "manifest.json").read_text())
        manifest["weights"][1]["start"] += 1  # open a gap in the NEW checkpoint
        (new / "manifest.json").write_text(json.dumps(manifest))
        with make_client(tmp_path, old, new) as client:
            resp = client.post("/diagnose", json={})
            assert resp.status_code == 422
            assert "gap" in resp.json()["detail"]

    def test_missing_dir_rejected_422(self, tmp_path):
        old, new = make_pair(tmp_path)
        with make_client(tmp_path, old, new) as client:
            resp = client.post("/diagnose", json={"old_dir": "/nonexistent"})
            assert resp.status_code == 422
            assert "not found" in resp.json()["detail"]


class TestReadFailure:
    def test_failed_read_returns_error_only_and_releases_slots(self, tmp_path, monkeypatch):
        old, new = make_pair(tmp_path)
        with make_client(tmp_path, old, new, read_concurrency=2) as client:
            gate = client.app.state.gate
            real_pread = os.pread
            calls = 0

            def flaky_pread(fd, n, offset):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise OSError("simulated disk failure")
                return real_pread(fd, n, offset)

            monkeypatch.setattr(os, "pread", flaky_pread)
            resp = client.post("/diagnose", json={})
            assert resp.status_code == 500
            body = resp.json()
            assert set(body.keys()) == {"detail"}  # error only, no partial counts
            assert "simulated disk failure" in body["detail"]
            # Every slot leaked by the failed scan would show up here.
            assert gate.available_slots == gate.max_slots
            monkeypatch.undo()
            # And the gate is still usable: a fresh diagnose succeeds.
            resp = client.post("/diagnose", json={})
            assert resp.status_code == 200

    def test_reshard_read_failure_releases_gate_slots(self, tmp_path, monkeypatch):
        from tests.conftest import write_source

        source = tmp_path / "src"
        write_source(source, total=500, step=1, weight_bounds=[0, 200, 500],
                     momentum_bounds=[0, 500])
        store = JobStore(tmp_path / "out")
        manifest = load_manifest(source)
        job = Job(
            job_id=compute_job_id(source, manifest, 2),
            source_dir=str(source),
            world_size=2,
            step=manifest.step,
            total_length=manifest.total_length,
            ranks=[RankState(rank=i, start=s, end=e)
                   for i, (s, e) in enumerate(rank_intervals(manifest.total_length, 2))],
        )
        store.add(job)
        gate = ReadGate(2)
        real_pread = os.pread
        calls = 0

        def flaky_pread(fd, n, offset):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("simulated source disk failure")
            return real_pread(fd, n, offset)

        monkeypatch.setattr(os, "pread", flaky_pread)
        job = execute_job(store, job, chunk_bytes=64, gate=gate)
        assert job.status == "failed"
        assert "simulated source disk failure" in job.error
        assert gate.available_slots == gate.max_slots


class RecordingGate(ReadGate):
    """ReadGate that observes concurrency: peak slots held and whether
    reshard and diagnose reads were ever in flight at the same time."""

    def __init__(self, max_slots):
        super().__init__(max_slots)
        self._lock = threading.Lock()
        self._active = {"reshard": 0, "diagnose": 0}
        self.peak = 0
        self.overlap = False

    @contextlib.contextmanager
    def slot(self, kind):
        with super().slot(kind):
            with self._lock:
                self._active[kind] += 1
                self.peak = max(self.peak, sum(self._active.values()))
                if all(self._active.values()):
                    self.overlap = True
            try:
                time.sleep(0.0002)  # stretch each read so the activities overlap
                yield
            finally:
                with self._lock:
                    self._active[kind] -= 1


class TestInterleavedReads:
    def test_reshard_and_diagnose_share_cap_and_stay_correct(self, tmp_path):
        total = 4000

        def mutate(w_old, m_old, w_new, m_new):
            w_new[11] = math.nan
            m_new[2222] = math.inf
            w_new[3000] = math.nan
            m_new[3000] = -math.inf

        old, new = make_pair(tmp_path, total=total, mutate=mutate)
        gate = RecordingGate(2)
        # sync_execution=False: the reshard runs in a background thread while
        # the diagnose request is served — both pull chunk reads from one gate.
        with make_client(tmp_path, old, new, chunk_bytes=32, sync=False,
                         gate=gate) as client:
            resp = client.post("/reshard", json={"new_world_size": 3})
            assert resp.status_code == 202, resp.text
            job_id = resp.json()["job_id"]

            d = client.post("/diagnose", json={})
            assert d.status_code == 200, d.text
            body = d.json()
            assert body["weights_only"] == {"count": 1, "first_indices": [11]}
            assert body["momentum_only"] == {"count": 1, "first_indices": [2222]}
            assert body["both"] == {"count": 1, "first_indices": [3000]}

            for _ in range(1000):
                status = client.get(f"/jobs/{job_id}").json()["status"]
                if status in {"done", "failed"}:
                    break
                time.sleep(0.01)
            assert status == "done"

        assert gate.peak <= 2  # shared cap never exceeded
        assert gate.overlap  # the two activities really interleaved
        assert gate.available_slots == 2  # nothing leaked
