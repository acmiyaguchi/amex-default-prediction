from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    files = Path(__file__).parent.parent.glob("venv/**/python.exe")
    for file in files:
        print(file)
        copyfile(file, file.parent / "python3.exe")
