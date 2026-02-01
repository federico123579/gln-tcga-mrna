# https://just.systems
alias s := sync
alias t := train
alias a := analyze
alias le := list-experiments

default:
    @just --list

sync:
    uv sync

train args="":
    uv run gln-train {{args}}

analyze args="":
    uv run gln-analyze {{args}}

experiment args="":
    uv run gln-experiment {{args}}

list-experiments:
    uv run gln-analyze --list-experiments
