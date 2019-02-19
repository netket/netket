import netket
import format
import inspect
import sys

if(len(sys.argv) != 2):
    print("Insert class name,for example: ")
    print("python3 make_class_docs.py netket.graph.Hypercube")
    exit()

assert("netket" in sys.argv[1])
class_name = eval(sys.argv[1])
markdown = format.format_class(class_name)
print(markdown)
