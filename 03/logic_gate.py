
import numpy as np
import sys

class LogicGate:
    def __init__(self):
        pass

    def and_gate(self, a, b):
        return np.logical_and(a, b)

    def nand_gate(self, a, b):
        return np.logical_not(np.logical_and(a, b))

    def or_gate(self, a, b):
        return np.logical_or(a, b)

    def nor_gate(self, a, b):
        return np.logical_not(np.logical_or(a, b))

    def xor_gate(self, a, b):
        return np.logical_xor(a, b)

    def help(self):
        help_message = """
        Usage of LogicGate class:
        
        - and_gate(a, b): Returns the AND operation between a and b.
        - nand_gate(a, b): Returns the NAND operation between a and b.
        - or_gate(a, b): Returns the OR operation between a and b.
        - nor_gate(a, b): Returns the NOR operation between a and b.
        - xor_gate(a, b): Returns the XOR operation between a and b.
        
        Note: Inputs should be arrays or scalars.
        """
        print(help_message)

# If the script is run directly, show help messages.
if __name__ == "__main__":
    gate = LogicGate()
    gate.help()
