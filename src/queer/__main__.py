"""``python -m queer``; the ``generator`` command starts without importing NumPy."""

import sys


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments[:1] == ["generator"]:
        from ._generator_boot import main as generator_main

        return generator_main(arguments[1:])
    from .cli import main as cli_main

    return cli_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
