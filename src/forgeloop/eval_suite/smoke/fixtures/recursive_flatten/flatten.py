def flatten(values):
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(value)
        else:
            result.append(value)
    return result
