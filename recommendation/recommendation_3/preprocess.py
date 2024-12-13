import re

def load_resources():
    with open('stopwords.txt', 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file.readlines()]

    cn_to_ko = {}
    with open('cn_to_ko.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            hanja, hangul = line.strip().split()
            cn_to_ko[hanja] = hangul

    symbol_replace = {}
    with open('symbol_replace.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            symbol, replacement = line.strip().split('\t')
            symbol_replace[symbol] = replacement

    with open('RegExp.txt', 'r', encoding='utf-8') as file:
        regex_patterns = [line.strip() for line in file.readlines()]

    with open('title_person.txt', 'r', encoding='utf-8') as file:
        people = [line.strip() for line in file.readlines()]

    return {
        'stopwords': stopwords,
        'cn_to_ko': cn_to_ko,
        'symbol_replace': symbol_replace,
        'regex_patterns': regex_patterns,
        'people': people
    }

def clean_text(text, stopwords, cn_to_ko, symbol_replace, regex_patterns, people):
    for stopword in stopwords:
        text = text.replace(stopword, '')

    for hanja, hangul in cn_to_ko.items():
        text = text.replace(hanja, hangul)

    for symbol, replacement in symbol_replace.items():
        text = text.replace(symbol, replacement)

    for pattern in regex_patterns:
        text = re.sub(pattern, '', text)

    for person in people:
        text = text.replace(person, '')

    text = re.sub(r'\s+', ' ', text)

    return text.strip()