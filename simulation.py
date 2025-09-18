import math, random, struct
from typing import Tuple, List, Optional

def factorial(n: int) -> int:
    return math.factorial(n)

def float_to_bits(f: float) -> int:
    """Convert float64 -> 64-bit int."""
    return struct.unpack('>Q', struct.pack('>d', f))[0]

def bits_to_float(b: int) -> float:
    """Convert 64-bit int -> float64."""
    return struct.unpack('>d', struct.pack('>Q', b))[0]

def bitflip_float(f: float) -> float:
    """Flip one random bit in the float64 representation."""
    if(f == 1):
        return 2
    b = float_to_bits(f)
    pos = random.randrange(64)   # choose random bit [0..63]
    b ^= (1 << pos)
    s = bits_to_float(b)
    return s

def taylor_exp_with_fault(
    x: float,
    n_terms: int = 15,
    fault_mode: Optional[str] = None,  # None | "skip" | "alter" | "bitflip"
    fault_index: Optional[int] = None, # which term to fault (0-based); if None choose randomly
    seed: Optional[int] = None
) -> Tuple[float, List[float], dict]:
    """
    Compute e^x via Taylor series with one injected fault.
    """
    if seed is not None:
        random.seed(seed)
    taylor_elements = []
    result = 0.0
    # pick fault index if needed
    if fault_mode is not None and fault_index is None:
        fault_index = random.randrange(n_terms)
    info = {"fault_was_injected": False, "fault_index": fault_index, "fault_mode": fault_mode,
            "original_value": None, "faulty_value": None}
    
    for n in range(n_terms):
        term = (x**n) / factorial(n)
        taylor_elements.append(term)
        if fault_mode is not None and n == fault_index:
            info["original_value"] = term
            if fault_mode == "skip":
                term = 0.0
            elif fault_mode == "alter":  # keep your old alter mode if needed
                delta = random.uniform(-0.5, 0.5)
                term = term * (1.0 + delta)
            elif fault_mode == "bitflip":
                term = bitflip_float(term)
            else:
                raise ValueError("fault_mode must be one of None, 'skip', 'alter', 'bitflip'")
            info["faulty_value"] = term
            info["fault_was_injected"] = True

        result += term

    return result, taylor_elements, info

def sigmoid(x,fault_mode):
    taylor_res,taylor_elements,info = taylor_exp_with_fault(-x,fault_mode=fault_mode)
    return 1/(1+taylor_res),taylor_elements,info

def check(y,x,taylor_elements,errors,info):
    left_side = (1/y - 1) + x
    right_side = sum(taylor_elements) - taylor_elements[1]
    if(left_side == right_side):
        errors += 1
    return errors

def fault_model_generation_report(samples,fault_mode):
    errors = 0
    for _ in range(samples):
        x = random.uniform(-5, 5)
        y,taylor_elements,info = sigmoid(x,fault_mode=fault_mode)  # 10 terms
        errors = check(y,x,taylor_elements,errors,info)
    print("Fault detection is {} in fault_mode {}".format((samples-errors)/samples,fault_mode))

samples = 1000000
fault_model_generation_report(samples,"skip")
fault_model_generation_report(samples,"alter")
fault_model_generation_report(samples,"bitflip")


# x = 5
# true_value = 1/(1+math.exp(-x))
# check(y,x,taylor_elements)
# print(f"Taylor approximation of e^{x}: {y}")
# print(f"Actual value: {true_value}")
# print(f"Error: {abs(y - true_value)}")