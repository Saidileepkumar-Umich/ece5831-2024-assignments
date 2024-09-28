
from logic_gate import LogicGate
import numpy as np

def test_logic_gates():
    gate = LogicGate()

    # Test data
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])

    print("AND Gate:", gate.and_gate(a, b))
    print("NAND Gate:", gate.nand_gate(a, b))
    print("OR Gate:", gate.or_gate(a, b))
    print("NOR Gate:", gate.nor_gate(a, b))
    print("XOR Gate:", gate.xor_gate(a, b))

if __name__ == "__main__":
    test_logic_gates()
