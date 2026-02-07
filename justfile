# https://just.systems
alias d := docs
alias s := sync
alias t := train
alias a := analyze
alias le := list-experiments

default:
    @just --list

docs:
    just -f documentation/justfile

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
