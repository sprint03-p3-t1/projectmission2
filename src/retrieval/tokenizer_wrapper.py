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

    def tokenize(self, text: str) -> List[str]:
        if self.mode == "kiwi":
            return [token.form for token in self.tokenizer.tokenize(text)]
        elif self.mode == "okt":
            return self.tokenizer.morphs(text)
