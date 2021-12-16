# -*- coding: utf-8 -*-
#===============================================================
#   Copyright (wifi) 2018 All rights reserved.
#
#   @filename: text_utils.py
#   @author: xxx@wifi.com
#   @date: 2018/01/13/ 14:47:50
#   @brief:
#
#   @history:
#
#================================================================

import sys
import unicodedata
import re
reload(sys)
sys.setdefaultencoding("utf-8")


SUPPORTED_CHARACTERS = {
    u",",  u"，", u".", u"。", u"?", u"？", u"《", u"》", u"!",
    u"！", u">", u"<", u"=", u"-", u"'", u"\"", u":", u"：",
    u"(", u"（", u")", u"）", u"[", u"]", u"【", u"】", " "
}


def is_alphabet(uchar):
    """whether this character is a letter
    """
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or \
            (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_number(uchar):
    """whether this character is a nummber
    """
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_chinese(uchar):
    """whether this character is chinese character
    """
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def uchar_type(uchar):
    """获取unicode字符类型
       0:中文,1:数字,2:字母,-1:其他符号
    """
    if is_chinese(uchar):
        return 0
    elif is_number(uchar):
        return 1
    elif is_alphabet(uchar):
        return 2
    else:
        return -1


def processTitle(text):
    out = []
    for x in text.lower().decode("utf-8", "ignore"):
        if uchar_type(x) == -1 and x not in SUPPORTED_CHARACTERS:
            continue
        out.append(x.encode("utf-8", "ignore"))
    text = "".join(out)

    text = text.replace(".mp4", "").replace(".mp3", "")
    text = re.sub(r'(?:用户)?[A-Za-z]?\d{5,}', "", text)
    return text


def strip_symbol(text):
    new_text = []
    text_unicode = text.decode("utf-8", "ignore")
    text_len = len(text_unicode)
    i = 0
    while i < text_len:
        if uchar_type(text_unicode[i]) != -1:
            break
        i += 1
    j = text_len - 1
    while j >= 0:
        if uchar_type(text_unicode[j]) != -1:
            break
        j -= 1
    if i > j:
        return ""
    return text_unicode[i:j+1].encode("utf-8", "ignore")


def count_chinese(text):
    """计算文本中汉字的数量
    """
    cnt = 0
    for x in text.decode("utf-8", "ignore"):
        if is_chinese(x):
            cnt += 1
    return cnt


def sbc_to_dbc(text):
    """全角转半角
    """
    text_unicode = text.decode("utf-8", "ignore")
    output_uchars = []
    for uchar in text_unicode:
        inside_code = ord(uchar)

        if inside_code == 12288:  # space is special
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        else:
            pass
        output_uchars.append(unichr(inside_code))

    return "".join(output_uchars).encode("utf-8", "ignore")


def remove_book_title_mark(text):
    if text.startswith("《") and text.endswith("》"):
        return text.decode("utf-8", "ignore")[1:-1].encode("utf-8", "ignore")
    return text


def is_numeric(text):
    try:
        float(text)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(text)
        return True
    except (TypeError, ValueError):
        pass

    return False


# python text_utils.py
if __name__ == "__main__":
    pass
