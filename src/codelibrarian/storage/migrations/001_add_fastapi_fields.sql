-- Migration to add FastAPI metadata fields to the symbols table
-- This allows codelibrarian to store and query the HTTP method and route for detected endpoints.

ALTER TABLE symbols ADD COLUMN http_method TEXT;
ALTER TABLE symbols ADD COLUMN route TEXT;
