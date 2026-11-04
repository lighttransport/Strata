"""A transport failure must not be described as a RAM failure."""
from pathlib import Path
import tempfile
import unittest
from serve.server import StrataEngine


class RemoteEngramFailure(unittest.TestCase):
    def note(self, log):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'engine.log'
            path.write_text(log)
            engine = object.__new__(StrataEngine)
            engine.log_path = str(path)
            engine.last_err = None
            engine.proc = None
            return engine.death_note()

    def test_failed_batch_identifies_remote_service(self):
        note = self.note('IB completion failure\nfatal: remote Engram batch failed; sequence discarded\n')
        self.assertIn('remote Engram service failed', note)
        self.assertNotIn('running out of RAM', note)

    def test_failed_heartbeat_identifies_remote_service(self):
        self.assertIn('remote Engram service failed', self.note('fatal: remote Engram heartbeat failed\n'))

    def test_historical_failure_does_not_override_a_later_session(self):
        note = self.note('fatal: remote Engram batch failed; sequence discarded\nnew engine started\nother failure\n')
        self.assertNotIn('remote Engram service failed', note)
