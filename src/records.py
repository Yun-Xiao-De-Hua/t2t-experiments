"""Append-only attempts and crash-safe commits for seven-category result records."""
import json
import os
from .config import CATEGORIES, canonical, digest, file_hash, save_new


class ResultStore:
    def __init__(self, root):
        self.root = root
        for name in CATEGORIES:
            (root / name).mkdir(parents=True, exist_ok=True)

    def append(self, category, record):
        path = self.root / CATEGORIES[category - 1] / 'records.jsonl'
        self.separate_partial_tail(path)
        with path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(canonical(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def separate_partial_tail(path):
        if path.exists() and path.stat().st_size:
            with path.open('rb') as f:
                f.seek(-1, 2)
                incomplete = f.read(1) != b'\n'
            if incomplete:
                with path.open('ab') as f:
                    f.write(b'\n')
                    f.flush()
                    os.fsync(f.fileno())

    def committed(self):
        path = self.root / 'commits.jsonl'
        if not path.exists():
            return {}
        result = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A truncated commit tail is NOT a completed sample.
                continue
            result[row['sample_id']] = row
        return result

    def commit(self, sample_id, attempt_id, records):
        keys = {'sample_id': sample_id, 'attempt_id': attempt_id, 'schema_version': 1}
        hashes = {}
        for category, value in records.items():
            payload = {**keys, **value}
            self.append(category, payload)
            hashes[str(category)] = digest(canonical(payload))
        self.separate_partial_tail(self.root / 'commits.jsonl')
        with (self.root / 'commits.jsonl').open('a', encoding='utf-8', newline='\n') as f:
            f.write(canonical({**keys, 'record_hashes': hashes}) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def checksums(self):
        # A new immutable checksum snapshot per invocation; old attempts are retained.
        import uuid
        save_new(self.root / f'SHA256SUMS_{uuid.uuid4().hex[:12]}', ''.join(
            f'{file_hash(p)}  {p.relative_to(self.root).as_posix()}\n'
            for p in sorted(self.root.rglob('*')) if p.is_file() and not p.name.startswith('SHA256SUMS')))
