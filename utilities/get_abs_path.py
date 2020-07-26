# %%
import os


def get_abs_path(currentfile, filepath):
    dirname = os.path.dirname(currentfile)
    return os.path.join(dirname, filepath)


if __name__ == "__main__":
    get_abs_path(__file__, "../sss.txt")
