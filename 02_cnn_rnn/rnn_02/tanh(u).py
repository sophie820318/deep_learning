import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(-6, 6, 1000)
y = np.tanh(u)

plt.figure(figsize=(6,4))
plt.plot(u, y, label='tanh(u)')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.ylim(-1.1, 1.1)
plt.xlim(-6, 6)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('tanh(u)')
plt.xlabel('u')
plt.ylabel('y')
plt.show()