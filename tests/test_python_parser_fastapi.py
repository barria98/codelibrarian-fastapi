"""Tests for Python parser's FastAPI endpoint detection."""

from pathlib import Path
import pytest
from codelibrarian.parsers.python_parser import PythonParser

FIXTURES = Path(__file__).parent / "fixtures"
FASTAPI_SAMPLE = FIXTURES / "fastapi_sample.py"


@pytest.fixture
def fastapi_result():
    parser = PythonParser()
    source = FASTAPI_SAMPLE.read_text()
    return parser.parse(FASTAPI_SAMPLE, source, "fastapi_sample")


def test_fastapi_endpoint_detection(fastapi_result):
    endpoints = [s for s in fastapi_result.symbols if s.kind == "fastapi_endpoint"]
    assert len(endpoints) == 1
    
    endpoint = endpoints[0]
    assert endpoint.name == "read_item"
    assert endpoint.http_method == "GET"
    assert endpoint.route == "/items/{item_id}"


def test_fastapi_call_graph_edge(fastapi_result):
    calls = fastapi_result.edges.calls
    # The endpoint should call its own underlying function
    has_edge = any(
        caller == "fastapi_sample.read_item" and callee == "read_item"
        for caller, callee in calls
    )
    assert has_edge, "Synthetic call edge for FastAPI endpoint is missing"


FASTAPI_ROUTER_SAMPLE = FIXTURES / "fastapi_router_sample.py"

@pytest.fixture
def fastapi_router_result():
    parser = PythonParser()
    source = FASTAPI_ROUTER_SAMPLE.read_text()
    return parser.parse(FASTAPI_ROUTER_SAMPLE, source, "fastapi_router_sample")


def test_fastapi_router_prefix(fastapi_router_result):
    endpoints = [s for s in fastapi_router_result.symbols if s.kind == "fastapi_endpoint"]
    assert len(endpoints) == 1
    
    endpoint = endpoints[0]
    assert endpoint.name == "read_user"
    assert endpoint.http_method == "GET"
    assert endpoint.route == "/users/{id}"
