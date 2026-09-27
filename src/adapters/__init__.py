"""学术搜索 API 适配层。"""
from src.adapters.base import AcademicSearchAdapter
from src.adapters.arxiv import ArxivAdapter
from src.adapters.crossref import CrossrefAdapter
from src.adapters.opencitations import OpenCitationsAdapter
from src.adapters.openalex import OpenAlexAdapter
from src.adapters.semantic_scholar import SemanticScholarAdapter

__all__ = [
    "AcademicSearchAdapter",
    "ArxivAdapter",
    "OpenAlexAdapter",
    "CrossrefAdapter",
    "OpenCitationsAdapter",
    "SemanticScholarAdapter",
]
