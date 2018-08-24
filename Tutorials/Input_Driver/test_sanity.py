import json

d1 = json.load(open("input.json"))
d2 = json.load(open("../Observables/sigmax.json"))

print(d1)
print(d2)
print(d1["Hamiltonian"] == d2["Hamiltonian"])
print(d1["Observables"] == d2["Observables"])
print(d1["GroundState"] == d2["GroundState"])

d3 = json.load(open("test.log"))
# d3["SigmaX"]
# print(d3)
for i in d3["Output"]:
    print(i)
    break
