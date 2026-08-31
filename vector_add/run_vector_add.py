import subprocess
import sys


def run_executable(executable):
    try:
        result = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

    except FileNotFoundError:
        print(f"Executable not found: {executable}")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("Program failed:")
        print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_vector_add.py <executable>")
        sys.exit(1)

    run_executable(sys.argv[1])