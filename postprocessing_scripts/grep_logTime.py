import re, pathlib
import numpy as np


def grep_logTime(filename, st):
    text = pathlib.Path(filename).read_text()
    pat = re.compile(rf"\b{re.escape(st)}\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    m = pat.search(text)
    n = float(m.group(1)) if m else None

    return n


def main():
    Time = []
    log = sorted(pathlib.Path(".").glob("log_*"))
    for filename in log:
        Time.append(grep_logTime(filename, 'Time'))

    return Time

if __name__ == "__main__":
    Time = np.array(main())
    print(Time)



