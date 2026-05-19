"""Shared utilities for sklearn-based train/predict scripts."""

def feat_to_dict(features):
    """Convert a list of feature strings to a dict.
    Features with '=' become key-value pairs; binary flags become key='1'."""
    d = {}
    for f in features:
        if '=' in f:
            k, v = f.split('=', 1)
            d[k] = v
        else:
            d[f] = '1'
    return d


def load_train_data(stream):
    """Read clf.feat format (tag\\tfeat1\\tfeat2\\t...) from stream.
    Returns (features, labels) as list of dicts and list of strings."""
    features, labels = [], []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        labels.append(parts[0])
        features.append(feat_to_dict(parts[1:]))
    return features, labels


def instances(stream):
    """Yield (feature_list, token_info_list) per sentence from devel/test feat file."""
    xseq, toks = [], []
    for line in stream:
        line = line.rstrip('\n')
        if not line:
            if xseq:
                yield xseq, toks
            xseq, toks = [], []
            continue
        fields = line.split('\t')
        toks.append(fields[:4])   # sid, form, start, end
        xseq.append(feat_to_dict(fields[5:]))
    if xseq:
        yield xseq, toks
