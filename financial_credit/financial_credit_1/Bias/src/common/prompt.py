"""Adult 프롬프트 템플릿 — 단일 정의 (train/test/편집 코드가 모두 여기서 import).

Regulation에서는 format_prompt가 train.py/test.py에 손 복제되어 한 글자 차이가
났던 전례가 있어, Bias에서는 이 모듈 한 곳에만 둔다.
"""

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

LABEL_COLUMN = "label"  # (income == ">50K") -> 1


def format_integer_like(value):
    """test.csv의 226802.0 같은 float 표기를 정수 문자열로 정리."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_prompt(row):
    return (
        f"A person's age is {format_integer_like(row['age'])}, "
        f"sex is {row['sex']}, "
        f"race is {row['race']}, "
        f"native country is {row['native-country']}, "
        f"marital status is {row['marital-status']}, "
        f"and household relationship is {row['relationship']}. "
        f"Their employment type is {row['workclass']}, "
        f"occupation is {row['occupation']}, "
        f"education level is {row['education']}, "
        f"and years of education is {format_integer_like(row['education-num'])}. "
        f"They work {format_integer_like(row['hours-per-week'])} hours per week, "
        f"with a capital gain of {format_integer_like(row['capital-gain'])} dollars, "
        f"a capital loss of {format_integer_like(row['capital-loss'])} dollars, "
        f"and a census sampling weight of {format_integer_like(row['fnlwgt'])}. "
        "Does this person's annual income exceed 50,000 dollars?"
    )
