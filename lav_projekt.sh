#!/bin/bash
# Laver projektmappen "numerisk" samme sted som denne fil ligger
cd "$(dirname "$0")"
mkdir -p numerisk/.vscode
cd numerisk

cat > requirements.txt << 'EOF'
numpy
scipy
matplotlib
sympy
ipykernel
EOF

cat > setup.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Faerdig! Environment er klar."
EOF
chmod +x setup.sh

cat > test_env.py << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

print("numpy version:", np.__version__)
x = np.linspace(0, 2 * np.pi, 200)
plt.plot(x, np.sin(x))
plt.title("Det virker!")
plt.show()
EOF

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "task.allowAutomaticTasks": "on"
}
EOF

cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Setup environment",
            "type": "shell",
            "command": "bash ./setup.sh",
            "runOptions": { "runOn": "folderOpen" },
            "presentation": { "reveal": "silent" }
        }
    ]
}
EOF

echo "Mappen er lavet her: $(pwd)"
echo "Installerer pakker (kan tage et par minutter)..."
bash ./setup.sh
command -v code >/dev/null && code .

