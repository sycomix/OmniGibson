# 1st batch: baselines
python tests/benchmark/profiling.py                  # baseline (fastest config possible)
python tests/benchmark/profiling.py -s Rs_int        # for vision research
python tests/benchmark/profiling.py -s Rs_int -r 1   # for basic robotics research
python tests/benchmark/profiling.py -s Rs_int -r 3   # for multi-agent research

# 2nd batch: compare different scenes
python tests/benchmark/profiling.py -r 1 -s house_single_floor
python tests/benchmark/profiling.py -r 1 -s grocery_store_cafe
python tests/benchmark/profiling.py -r 1 -s Pomaria_0_garden

# 3rd batch: OG non-physics features
python tests/benchmark/profiling.py -r 1 -s Rs_int -w             # fluids (water)
python tests/benchmark/profiling.py -r 1 -s Rs_int -c             # soft body (cloth)
python tests/benchmark/profiling.py -r 1 -s Rs_int -p             # macro particle system (diced objects)
python tests/benchmark/profiling.py -r 1 -s Rs_int -w -c -p       # everything