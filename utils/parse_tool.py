import json


def common_parse(text: str, default=''):
    """通用解析函数"""
    if not text or not isinstance(text, str):
        return default if default is not None else text
    cur_text = text.strip()
    try:
        return json.loads(cur_text)
    except Exception as e:
        print(f'common_parse error for text: {text[:50]}...', e)
        return default if default is not None else text

def common_dumps(object):
    return json.dumps(object)
