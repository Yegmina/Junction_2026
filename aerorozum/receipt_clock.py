"""Conservative selection outside original receipt-clock rollback windows."""


def rollback_windows(sequences,guard_ns=2_000_000_000):
    if type(guard_ns) is not int or guard_ns<2_000_000_000:
        raise ValueError('Preserve at least the full two-second freshness guard')
    intervals=[]
    for times in sequences:
        previous=None
        for stamp in times:
            if type(stamp) is not int or stamp<=0:
                raise ValueError('Original positive integer receipt timestamps required')
            if previous is not None and stamp<previous:
                intervals.append((stamp-guard_ns,previous+guard_ns))
            previous=stamp
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    return merged


def safe_native_indices(local_records,native_count,clock_sequences):
    if len(local_records)!=native_count:
        raise ValueError('Receipt/native frame count mismatch; clock isolation cannot repair frame identity')
    if not all(type(r.get('frame')) is int for r in local_records):
        raise ValueError('Explicit original frame counters required')
    if any(b['frame']!=a['frame']+1 for a,b in zip(local_records,local_records[1:])):
        raise ValueError('Contiguous frame counters required')
    times=[r['received_at_ns'] for r in local_records]
    windows=rollback_windows([times,*clock_sequences])
    selected=[];last=None
    for i,stamp in enumerate(times):
        if any(lo<=stamp<=hi for lo,hi in windows):continue
        # Equal timestamps cannot contribute two source-clock samples.
        if last is not None and stamp<=last:continue
        selected.append(i);last=stamp
    return selected,dict(rollback_windows_ns=windows,guard_s=2,
        total_native_frames=native_count,unambiguous_frames=len(selected),
        excluded_frames=native_count-len(selected),timestamps_modified=False,
        frame_counters_modified=False,frame_identity_binding='Native count matches explicit contiguous receipt counters one-to-one')
