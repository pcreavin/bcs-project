# Activating the virtual environment (Windows)

This project uses a standard Python venv located at `./venv`. Below are three convenient options to create and activate it on Windows.

1) One-time temporary bypass (PowerShell)

Run this in the VS Code PowerShell integrated terminal. It will bypass execution policy just for this session and activate the venv:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; .\venv\Scripts\Activate.ps1
```

2) Persistent for your user (recommended)

Run once in PowerShell (no admin required). This allows locally-signed and local scripts to run for your user account.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\venv\Scripts\Activate.ps1
```

3) Use Command Prompt (no PowerShell policy changes)

Change the integrated terminal to Command Prompt in VS Code (this repository includes `.vscode/settings.json` to default to Command Prompt). Then run:

```cmd
venv\Scripts\activate
```

Helper files in the repository
- `scripts/setup_venv.ps1` — creates the venv if missing and prints activation options.
- `scripts/activate_cmd.bat` — helper batch file to activate the venv from CMD.
- `.vscode/settings.json` — workspace settings that default terminal to Command Prompt and point Python interpreter to the venv.

Installing dependencies

After activation, install dependencies with:

```powershell
pip install -r requirements.txt
```

If you still get an execution policy error, copy-paste the exact error message here and I'll help you troubleshoot further.
