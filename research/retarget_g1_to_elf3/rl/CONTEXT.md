# ELF3 Motion Policy Context

This context defines the language used to discuss ELF3 motion tracking policies and reference motions.

## Language

**Reference Motion**:
A target motion trajectory that the robot policy attempts to follow.
_Avoid_: demo, sample, NPZ

**Training Motion**:
A reference motion included in skill policy training.
_Avoid_: baseline clip, debug clip

**Benchmark Evaluation**:
An evaluation set used to compare skill policy quality across training runs.
_Avoid_: smoke test, demo render

**Debug Evaluation**:
An evaluation run used for smoke tests, regressions, or visual debugging.
_Avoid_: benchmark, score

**Motion Category**:
A named family of reference motions with similar balance and coordination demands.
_Avoid_: folder, dataset split

**Skill Policy**:
A policy responsible for tracking one coherent motion skill or a closely related set of motion categories.
_Avoid_: category policy, sub-policy, expert

**Policy Set**:
A group of skill policies that together cover the desired reference motions.
_Avoid_: model zoo, policy folder

**Policy Route**:
An explicit assignment from a reference motion to the skill policy that should track it.
_Avoid_: classifier, gate

**Reference Horizon**:
The future span of a reference motion made visible to a skill policy.
_Avoid_: lookahead hack, future frames

**Root-Relative Reference**:
A reference target expressed relative to the robot root state rather than an absolute world origin.
_Avoid_: world target, global trajectory

**Universal Tracking Policy**:
A single policy expected to track all reference motions across all motion categories.
_Avoid_: one big model, general model

**ELF3 RL Runtime Environment**:
An isolated runtime context used to train, evaluate, and export ELF3 motion tracking policies.
_Avoid_: base environment, system Python, shared driver setup

## Relationships

- A **Reference Motion** belongs to exactly one **Motion Category**.
- A **Training Motion** is a **Reference Motion** from the classified training set.
- A **Benchmark Evaluation** excludes legacy root-level baseline clips.
- A **Debug Evaluation** may use legacy root-level baseline clips.
- A **Skill Policy** covers one or more **Motion Categories**.
- A **Policy Route** selects exactly one **Skill Policy** for a **Reference Motion** by explicit motion identity.
- A **Skill Policy** may use a **Reference Horizon** instead of only the current reference frame.
- A **Reference Horizon** uses **Root-Relative Reference** for root displacement.
- The initial **Policy Set** has three **Skill Policies**: locomotion, posture balance, and upper-body.
- The **Policy Set** does not have to match the source data directory layout.
- A **Universal Tracking Policy** covers all **Motion Categories**.
- An **ELF3 RL Runtime Environment** supports **Debug Evaluation**, training, and policy export without changing shared machine drivers.

## Example Dialogue

> **Dev:** "Should one **Universal Tracking Policy** handle walking, single-leg balance, and upper-body gestures?"
> **Domain expert:** "No. Use **Skill Policies** so each policy handles a coherent motion family."

## Flagged Ambiguities

- "network architecture" was used to mean both policy capacity and policy decomposition; resolved: policy decomposition can use multiple **Skill Policies**.
- "category" was used as if it must define policy boundaries; resolved: **Skill Policy** boundaries are based on motion dynamics, not source data directories.
- "routing" was resolved as explicit **Policy Routes** by motion identity, not directory names, keyword rules, an automatic classifier, or a learned gate.
