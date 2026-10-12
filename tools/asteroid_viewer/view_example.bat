@echo off
rem Opens the full period_search input/output; use Compute shape for the shape model.
set PS=%~dp0..\..\period_search
start "" "%~dp0bin\asteroid_viewer.exe" "%PS%\period_search_in" "%PS%\period_search_out"
