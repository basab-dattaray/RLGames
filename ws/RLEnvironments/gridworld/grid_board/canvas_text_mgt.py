def canvas_text_mgt(canvas):
    _dict = {}

    def fn_push(key, val):
        nonlocal _dict

        if key in _dict:
            lst_of_refs = _dict.pop(key)
            for ref in lst_of_refs:
                canvas.delete(ref)

        _dict[key] = val

    return fn_push
