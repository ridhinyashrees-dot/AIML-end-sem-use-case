@echo off
python -m venv venv || (echo Python not found & exit /b 1)
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Next: 1) edit .env (EE_PROJECT)  2) run: earthengine authenticate  3) put GSI CSV in data\raw\manganese_deposits.csv  4) run.bat
