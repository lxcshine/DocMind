"""Smoke test for the new StateDB layer (isolated, no package init)."""
import importlib.util
import os
import sys
import tempfile


def _load_module_from_path(name: str, path: str):
    """Load a single .py file as a module without triggering package init."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_state_db():
    here = os.path.dirname(os.path.abspath(__file__))
    state_db = _load_module_from_path(
        "infrastructure.state_db_isolated",
        os.path.join(here, "infrastructure", "state_db.py"),
    )
    StateDB = state_db.StateDB

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        db = StateDB(db_path)

        # Insert and read back
        db.execute(
            "INSERT INTO rate_limit_buckets (client_ip, ts) VALUES (?, ?)",
            ("127.0.0.1", 1.0),
        )
        rows = db.query_all("SELECT * FROM rate_limit_buckets")
        assert len(rows) == 1
        assert rows[0]["client_ip"] == "127.0.0.1"

        # Multi-process: a second instance should see the same data.
        db2 = StateDB(db_path)
        doc_check = db2.query_one(
            "SELECT 1 FROM rate_limit_buckets WHERE client_ip = ?",
            ("127.0.0.1",),
        )
        assert doc_check is not None, "WAL mode failed - data not visible to second connection"
        print("OK: state_db basic CRUD + multi-process visibility work")

        # Documents table
        db.execute(
            """INSERT INTO documents (doc_id, filename, file_type, file_size, sections_count, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("abc", "test.pdf", "PDF", 1024, 5, "completed", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        doc = db.query_one("SELECT * FROM documents WHERE doc_id = ?", ("abc",))
        assert doc is not None
        assert doc["filename"] == "test.pdf"

        # Progress table
        db.execute(
            """INSERT INTO progress (doc_id, filename, status, progress, current_step, total_steps, completed_steps, sections_count, error, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("abc", "test.pdf", "processing", 50, "step 1", 3, 1, 0, None, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        prog = db.query_one("SELECT * FROM progress WHERE doc_id = ?", ("abc",))
        assert prog["status"] == "processing"
        assert prog["progress"] == 50
        print("OK: state_db documents & progress tables work")

        # Transaction rollback test
        with pytest_raises(RuntimeError):
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO documents (doc_id, filename, file_type, file_size, sections_count, status, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("xyz", "x.pdf", "PDF", 100, 0, "uploaded", "2026", "2026"),
                )
                raise RuntimeError("simulated failure")
        # Should have been rolled back
        x = db.query_one("SELECT 1 FROM documents WHERE doc_id = ?", ("xyz",))
        assert x is None, "transaction rollback did not work"
        print("OK: state_db transaction rollback works")


class pytest_raises:
    """Minimal context manager that mirrors pytest.raises (no extra deps)."""
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} but no exception was raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        return True


if __name__ == "__main__":
    test_state_db()
    print("\nALL SMOKE TESTS PASSED")
