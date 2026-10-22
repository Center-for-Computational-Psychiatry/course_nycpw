# Tutorial 2: Introduction to Reinforcement Learning — Quick Start

New York Computational Psychiatry Workshop · 2026 · Angela Radulescu

## 1. Download the folder

Open the Dropbox link, click **Download**, and unzip the file. Keep everything in the folder together (the notebook needs `custom.css` and `public/`).

## 2. Install uv (one time)

**Mac / Linux** — open Terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** — open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then **close the window and open a new one**.

## 3. Open the tutorial

In the new window, go to the unzipped folder. Type `cd ` (with a space), drag the folder into the window, and press Enter. Then run:

```bash
uvx marimo edit --sandbox tutorial02_RL.py
```

The notebook opens in your browser. The first launch takes a minute or two to download what it needs.

## If something goes wrong

- **`uvx: command not found`** — open a new terminal window and try again.
- **Blank notebook or missing images** — you're not in the right folder. Run `cd` again (step 3) and make sure the folder contains `tutorial02_RL.py`.
- **Anything else** — press `Ctrl-C` in the terminal, then run the command from step 3 again.
