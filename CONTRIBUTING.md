# Contributing guidelines

The full contributing guide has been moved to our website. Please visit

https://netket.readthedocs.io/en/latest/developer-guides/contributing.html

## AI-assisted contributions

AI tools are welcome, but a contribution should be worth more to the project than the time it
takes to review it: an unreviewed wall of LLM output costs us more to read and correct than
we'd spend fixing it ourselves. You are the author: you must be able to explain any change or
bug report in your own words.

**Human-written framing.** If you don't yet an estabilished track record of merged PRs here,
every issue/PR must open with a short paragraph manually written by the human, not generated or reprocessed by an AI.
Additional AI-generated text is welcome below it, collapsed in a `<details>` block:

```markdown
I was trying to run some simulations on more than 1 node with MPI, but the simulation kept
getting deadlocks. I instructed my favourite LLM to look into this and we realised that there
is a race in `MCState.sample()` under MPI.
This PR fixes it.

<details><summary>AI-assisted details</summary>
...
</details>
```

**The human must manually curate the AI-generated text:** How much you must curate that collapsed text scales with the change: a one-line human summary
is enough for a small bugfix; larger or deeper PRs need the AI text itself edited for
accuracy, compactness and clarity. They cannot be pasted raw.

**Discuss before you build.** New features and anything requiring a design decision need a
discussed, accepted issue before a PR. Small, self-contained bugfixes don't. Bugfixes that
require a design choice (API or semantics changes) do, for the same reason.
