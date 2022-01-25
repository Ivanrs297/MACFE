import subprocess
from datetime import datetime
import time
import os

print("STARTING GRIDSEARCH...")
print(f"[{datetime.now()}]")
start = time.time()

# Depth Iterations
d_list  = [1, 2, 3, 4]  
d_list = [str(d) for d in d_list]
d_list = " ".join(d_list)

# Selection Threshold
s_list  = [.2, .4, .6, .8, 1]   # top 20%, top 40%, ...
s_list = [str(d) for d in s_list]
s_list = " ".join(s_list)

current_abs_path = os.path.abspath(os.getcwd())

subprocess.call(f'python {current_abs_path}\code\experiments.py -d_l {d_list} -s_l {s_list}',shell=True)

end = time.time()-start
print(f"[{datetime.now()}]")
print(f"\nTotal time = {int(end//3600):02d}:{int((end//60))%60:02d}:{end%60:.6f}")