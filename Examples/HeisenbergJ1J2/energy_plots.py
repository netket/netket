import json
import matplotlib.pyplot as plt

gs_data = json.load(open("ground_state.log"))
es_data = json.load(open("excited_state.log"))

plt.plot(range(200), gs_data["Energy"]["Mean"]["real"])
plt.plot(range(200, 250), es_data["Energy"]["Mean"]["real"])
plt.show()

# Now zoom in on the low-energy end and compare to the energy found in 2005.14142
best = -0.497629 * 400
plt.plot(range(200), gs_data["Energy"]["Mean"]["real"])
plt.plot(range(200, 250), es_data["Energy"]["Mean"]["real"])
plt.plot([0, 250], [best, best])
plt.ylim(-199.5, -196)
plt.show()
