# Three Skill Policies With Explicit Routing

Accepted. ELF3 motion tracking will use a policy set with three independently trained skill policies instead of one universal tracking policy: `locomotion_policy`, `posture_balance_policy`, and `upper_body_policy`. The universal policy path is simpler to train and deploy on paper, but it concentrates very different balance, locomotion, and upper-body coordination demands into one model; independent skill policies keep failures local while preserving a common `observation -> action` interface.

Each reference motion is assigned to exactly one skill policy through an explicit policy route keyed by motion identity. Directory names, keyword rules, learned gates, and automatic classifiers are not the routing source of truth in the first version, because compound motions cross source-directory boundaries and need auditable routing decisions.
