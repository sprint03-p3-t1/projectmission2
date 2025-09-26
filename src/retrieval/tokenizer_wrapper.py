from typing import List

class TokenizerWrapper:
    def __init__(self, engine="kiwi"):
        if engine == "kiwi":
            from kiwipiepy import Kiwi
            self.tokenizer = Kiwi()
            self.mode = "kiwi"
        elif engine == "okt":
            from konlpy.tag import Okt
            self.tokenizer = Okt()
            self.mode = "okt"
        else:
            raise ValueError(f"지원되지 않는 분석기: {engine}")

    def tokenize_korean(self, text: str, use_bigrams: bool = True) -> List[str]:
        # ✅ 명사만 추출
        if self.mode == "kiwi":
            tokens = [token.form for token in self.tokenizer.tokenize(text) if token.tag.startswith("NN")]
        elif self.mode == "okt":
            tokens = self.tokenizer.nouns(text)

        # ✅ 불용어 제거
        stopwords = {"에서", "는", "은", "이", "가", "하", "어야", "에", "을", "를", "도", "로", "과", "와", "의", "?", "다"}
        tokens = [t for t in tokens if t not in stopwords]

        # ✅ 바이그램 추가
        if use_bigrams:
            bigrams = [tokens[i] + tokens[i+1] for i in range(len(tokens) - 1)]
            tokens += bigrams

        return tokens

