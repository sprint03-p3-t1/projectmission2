class RetrieverError(Exception):
    """Retriever에서 발생하는 예외를 위한 기본 클래스"""
    pass

class ChunkLoadingError(RetrieverError):
    """JSON 청크 로딩 실패 시 발생하는 예외"""
    pass
