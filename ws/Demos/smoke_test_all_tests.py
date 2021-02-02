import os

from ws.RLUtils.setup.exec_mgt import exec_mgt

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    fn_traverse_dir, fn_stats = exec_mgt(__file__)
    fn_traverse_dir(cwd)
    total_count, failures = fn_stats()
    print(f'total_count: {total_count}, failures: {failures}')





