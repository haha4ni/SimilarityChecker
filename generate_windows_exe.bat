@echo off
echo "Start transfer py to exe..."
set OUT_PATH=.\tmp_out
mkdir %OUT_PATH%
copy .\CheckMonitor.py %OUT_PATH%
cd %OUT_PATH%
pyinstaller -F .\CheckMonitor.py
echo "Windows exeuatable generated."
cd ..\
mkdir .\exe_out
copy %OUT_PATH%\dist\CheckMonitor.exe .\exe_out
copy .\config.json .\exe_out
echo "Please check exe_out folder for the output files."
rmdir /S /Q %OUT_PATH%
pause