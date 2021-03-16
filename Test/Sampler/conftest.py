import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--sampler",
        action="store",
        default="",
        help="sampler: exact, metropolis, metropolispt or pt, local, hamiltonian, custom",
    )

    parser.addoption(
        "--mpow",
        action="store",
        default="single",
        help="mpow: single, all, 1,2,3...",
    )
