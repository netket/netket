# Copyright 2025 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Command-line interface for the citation system.
"""

import argparse

from .core import cite


def cite_cli() -> None:
    """
    Command-line interface for the citation system.

    This function is called by the 'netket.cite' console script.
    """
    parser = argparse.ArgumentParser(
        description="Generate citation information for NetKet and associated algorithms",
        prog="netket.cite",
    )
    parser.add_argument(
        "--bib",
        nargs="?",
        const="references.bib",
        help="Generate a BibTeX file. Optionally specify filename (default: references.bib)",
    )

    args = parser.parse_args()

    # Import NetKet to register all citations from decorated functions
    import netket  # noqa: F401

    if args.bib:
        # Automatically add .bib extension if not present (case-insensitive)
        filename = args.bib
        if not filename.lower().endswith(".bib"):
            filename += ".bib"
        cite(output_file=filename, from_cli=True)
    else:
        cite(from_cli=True)
