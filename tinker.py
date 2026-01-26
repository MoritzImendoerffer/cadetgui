from CADETProcess.simulator import Cadet

simulator = Cadet()
print(f"Before check_cadet: {simulator.cadet_path}")

try:
    simulator.check_cadet()
    print(f"After check_cadet: {simulator.cadet_path}")
except Exception as e:
    print(f"Exception: {e}")