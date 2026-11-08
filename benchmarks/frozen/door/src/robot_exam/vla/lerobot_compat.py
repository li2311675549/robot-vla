"""Process-local HF feature compatibility; never edits installed packages."""


def enable_list_feature_compat():
    # datasets 4 writes scalar-array metadata as List. The existing OpenPI
    # snapshot uses datasets 3, whose equivalent scalar-array feature is
    # Sequence. Preserve Arrow payloads; enable only the missing reader name.
    from datasets.features import features
    if 'List' not in features._FEATURE_TYPES:
        features._FEATURE_TYPES['List']=features.Sequence
        return 'List -> Sequence reader alias (process-local)'
    return 'native List feature available'
