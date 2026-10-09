"""``python -m tradingbot`` : délègue à :func:`tradingbot.cli.main`."""

from __future__ import annotations

import sys

from tradingbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
