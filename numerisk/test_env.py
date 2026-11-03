import numpy as np
import matplotlib.pyplot as plt

print("numpy version:", np.__version__)
x = np.linspace(0, 2 * np.pi, 200)
plt.plot(x, np.sin(x))
plt.title("Det virker!")
plt.show()
