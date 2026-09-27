from .workflow import main
import sys

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("ERROR:", str(exc), file=sys.stderr)
        sys.exit(2)
