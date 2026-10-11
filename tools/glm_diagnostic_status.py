"""Read persisted GLM task/phase logs, including runs left unfinished by reboot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def rows(path):
    if not path.exists():
        return [], 0
    valid, malformed = [], 0
    for line in path.read_text(errors='replace').splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('expected object')
            valid.append(row)
        except ValueError:
            malformed += 1
    return valid, malformed


def inspect(directory, boot_id=None):
    boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    campaign, errors = rows(directory/'campaign.tasks.jsonl')
    runs = []
    for path in sorted(directory.rglob('*.guard.json')):
        guard = json.loads(path.read_text())
        name = str(path.relative_to(directory)).removesuffix('.guard.json')
        prefix = path.parent/path.name.removesuffix('.guard.json')
        events, event_errors = rows(prefix.with_suffix('.tasks.jsonl'))
        phases, phase_errors = rows(prefix.with_suffix('.engine.jsonl'))
        finished = 'exit_code' in guard
        state = ('passed' if guard.get('passed') is True and finished else
                 'failed' if finished else
                 'unfinished across reboot' if guard.get('boot_id') != boot_id else
                 'unfinished in current boot')
        row = dict(task=name, state=state, boot_id=guard.get('boot_id'),
                   last_event=events[-1] if events else None,
                   last_engine_phase=phases[-1] if phases else None,
                   exit_code=guard.get('exit_code'), faults=guard.get('faults', []),
                   termination_reason=guard.get('termination_reason'),
                   malformed_log_lines=event_errors+phase_errors)
        if phases and 'unix_ms' in phases[-1]:
            row['last_phase_utc'] = datetime.fromtimestamp(phases[-1]['unix_ms']/1000,
                                                         timezone.utc).isoformat()
        runs.append(row)
    return dict(current_boot_id=boot_id, last_campaign_event=campaign[-1] if campaign else None,
                malformed_campaign_lines=errors, runs=runs,
                note='An unfinished run does not identify the cause of a reboot. Non-synchronized phase markers record enqueue progress.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory), indent=2))


if __name__ == '__main__':
    main()
