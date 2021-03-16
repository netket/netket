def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)
