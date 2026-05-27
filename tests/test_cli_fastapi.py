from click.testing import CliRunner
from codelibrarian.cli import main
from codelibrarian.models import SymbolRecord, SearchResult
from unittest.mock import patch, MagicMock

def test_search_fastapi_only():
    sym1 = SymbolRecord(
        id=1, file_id=1,
        name="read_item",
        qualified_name="read_item",
        kind="fastapi_endpoint",
        file_path="main.py",
        relative_path="main.py",
        line_start=1,
        line_end=2,
        signature=None, docstring=None, parameters=[], return_type=None, decorators=[], parent_id=None,
        http_method="GET",
        route="/items/{item_id}"
    )
    sym2 = SymbolRecord(
        id=2, file_id=1,
        name="some_func",
        qualified_name="some_func",
        kind="function",
        file_path="main.py",
        relative_path="main.py",
        line_start=5,
        line_end=6,
        signature=None, docstring=None, parameters=[], return_type=None, decorators=[], parent_id=None,
    )
    
    with patch("codelibrarian.cli.Config.load_from_cwd") as mock_config, \
         patch("codelibrarian.storage.store.SQLiteStore") as mock_store, \
         patch("codelibrarian.searcher.Searcher.search") as mock_search, \
         patch("pathlib.Path.exists", return_value=True):
         
        mock_config.return_value.db_path.exists.return_value = True
        mock_search.return_value = [
            SearchResult(symbol=sym1, score=1.0, match_type="fulltext"),
            SearchResult(symbol=sym2, score=0.8, match_type="fulltext")
        ]
        
        runner = CliRunner()
        result = runner.invoke(main, ["search", "query", "--fastapi-only"])
        
        assert result.exit_code == 0
        assert "some_func" not in result.output
        assert "=== FastAPI Endpoints ===" in result.output
        assert "read_item" in result.output
        assert "GET" in result.output
        assert "/items/{item_id}" in result.output


def test_search_grouping_format():
    sym1 = SymbolRecord(
        id=1, file_id=1,
        name="read_item",
        qualified_name="read_item",
        kind="fastapi_endpoint",
        file_path="main.py",
        relative_path="main.py",
        line_start=1,
        line_end=2,
        signature=None, docstring=None, parameters=[], return_type=None, decorators=[], parent_id=None,
        http_method="GET",
        route="/items/{item_id}"
    )
    sym2 = SymbolRecord(
        id=2, file_id=1,
        name="some_func",
        qualified_name="some_func",
        kind="function",
        file_path="main.py",
        relative_path="main.py",
        line_start=5,
        line_end=6,
        signature=None, docstring=None, parameters=[], return_type=None, decorators=[], parent_id=None,
    )
    
    with patch("codelibrarian.cli.Config.load_from_cwd") as mock_config, \
         patch("codelibrarian.storage.store.SQLiteStore") as mock_store, \
         patch("codelibrarian.searcher.Searcher.search") as mock_search, \
         patch("pathlib.Path.exists", return_value=True):
         
        mock_config.return_value.db_path.exists.return_value = True
        mock_search.return_value = [
            SearchResult(symbol=sym1, score=1.0, match_type="fulltext"),
            SearchResult(symbol=sym2, score=0.8, match_type="fulltext")
        ]
        
        runner = CliRunner()
        result = runner.invoke(main, ["search", "query"])
        
        assert result.exit_code == 0
        assert "=== FastAPI Endpoints ===" in result.output
        assert "read_item" in result.output
        assert "GET" in result.output
        assert "/items/{item_id}" in result.output
        
        assert "some_func" in result.output
        assert "Location" in result.output


def test_lookup_fastapi():
    sym1 = SymbolRecord(
        id=1, file_id=1,
        name="read_item",
        qualified_name="read_item",
        kind="fastapi_endpoint",
        file_path="main.py",
        relative_path="main.py",
        line_start=1,
        line_end=2,
        signature=None, docstring=None, parameters=[], return_type=None, decorators=[], parent_id=None,
        http_method="GET",
        route="/items/{item_id}"
    )
    
    with patch("codelibrarian.cli.Config.load_from_cwd") as mock_config, \
         patch("codelibrarian.storage.store.SQLiteStore") as mock_store, \
         patch("codelibrarian.searcher.Searcher.lookup_symbol") as mock_lookup, \
         patch("pathlib.Path.exists", return_value=True):
         
        mock_config.return_value.db_path.exists.return_value = True
        mock_lookup.return_value = [sym1]
        
        runner = CliRunner()
        result = runner.invoke(main, ["lookup", "read_item"])
        
        assert result.exit_code == 0
        assert "Method:    GET" in result.output
        assert "Route:     /items/{item_id}" in result.output
