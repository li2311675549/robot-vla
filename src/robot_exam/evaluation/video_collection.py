"""Select a bounded demonstration collection without changing its trial ledger."""
from collections import Counter


CATEGORIES = ('apple', 'banana', 'orange')


def select_grasp_videos(rows, total=40, successes_per_category=8, minimum_failures=4):
    """Return display IDs, or None while quotas are unmet.

    Eligibility and success come from the unchanged physical evaluator. This
    subset is a display collection, never the denominator of a success rate.
    Prefer different scenes/noise before repeated identical initialization.
    """
    if total < successes_per_category*len(CATEGORIES)+minimum_failures:
        raise ValueError('Display quotas exceed collection size')
    eligible = [r for r in rows if r.get('exhibit_eligible')]
    if len({r['id'] for r in eligible}) != len(eligible):
        raise ValueError('A physical attempt cannot supply multiple display entries')
    if len(eligible) < total:
        return None
    chosen, seen = [], set()
    def add(row):
        chosen.append(row); seen.add(row['id'])
    for category in CATEGORIES:
        group = [r for r in eligible if r['category'] == category and r['success']]
        if len(group) < successes_per_category:
            return None
        unique, repeats, contexts = [], [], set()
        for row in group:
            key = (row['scene_seed'], row['policy_noise_seed'])
            (repeats if key in contexts else unique).append(row)
            contexts.add(key)
        for row in (unique+repeats)[:successes_per_category]:
            add(row)
    failures = [r for r in eligible if not r['success']]
    if len(failures) < minimum_failures:
        return None
    unique, repeats, kinds = [], [], set()
    for row in failures:
        key = (row['category'], row.get('failure_category'), row['scene_seed'])
        (repeats if key in kinds else unique).append(row); kinds.add(key)
    for row in (unique+repeats)[:minimum_failures]:
        add(row)
    remaining = [r for r in eligible if r['id'] not in seen]
    while len(chosen) < total:
        counts = Counter(r['category'] for r in chosen)
        row = min(remaining, key=lambda r:(not r['success'], counts[r['category']],
                                         r['policy_noise_seed'] == 0, r['id']))
        add(row);remaining.remove(row)
    return [r['id'] for r in chosen]
