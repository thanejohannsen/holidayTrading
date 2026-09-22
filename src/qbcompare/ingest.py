"""Attribute plays to a quarterback by parsing ``play_text``.

The bulk feed ships a ``passer_player_name`` column, but it is mangled on roughly
two thirds of rows whenever the play text uses the verbose NCAA scorer format
(it emits things like ``"o Huddle-Shotgun #16 A.Manning"``). So attribution is
done here from the text itself.

The text arrives in two formats and both must be handled:

    terse    "Arch Manning pass complete to Jack Endries for 3 yds to the TEX 34"
    verbose  "(01:57) No Huddle-Shotgun #16 A.Manning pass complete short middle
              to #0 D.Moore Jr. caught at TEX26, for 7 yards"

A quarterback's name can appear in a play he did not initiate — as the receiver
of a lateral, or in a tackle credit. So a match only counts when the name is
followed by an action verb, i.e. when the name sits in the *actor slot*.
"""

from __future__ import annotations

import re

import pandas as pd

# Verbs that mark the preceding name as the player who initiated the play.
ACTION = r"(?:pass|passes|rush|rushes|run|runs|scramble[sd]?|sacked|kneel[sd]?|spike[sd]?)"

QUARTERBACKS = {
    "Manning": {
        "team": "Texas",
        # Terse uses the full name; verbose uses first-initial-dot-surname.
        "patterns": [r"Arch\s+Manning", r"A\.\s?Manning"],
    },
    "Stockton": {
        "team": "Georgia",
        "patterns": [r"Gunner\s+Stockton", r"G\.\s?Stockton"],
    },
}

# play_type values that represent a pass attempt by the quarterback.
PASS_TYPES = {
    "Pass Reception",
    "Pass Completion",
    "Pass Incompletion",
    "Passing Touchdown",
    "Interception Return",
    "Interception Return Touchdown",
    "Interception",
    "Pass Interception Return",
    "Pass Interception",
}
SACK_TYPES = {"Sack", "Sack Touchdown"}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}


def _actor_regex(patterns: list[str]) -> re.Pattern:
    """Match the QB's name only when an action verb follows it."""
    name = "|".join(patterns)
    return re.compile(rf"(?:{name})\s+{ACTION}\b", re.IGNORECASE)


def _reverse_regex(patterns: list[str]) -> re.Pattern:
    """Match the scoring-summary format, where the passer comes last.

    Touchdown passes are sometimes written as "Kaliq Lockett 30 Yd pass from
    Arch Manning (Mason Shipley Kick)". These are real snaps, not duplicates of
    another row, so missing them would drop five of Manning's touchdowns.
    """
    name = "|".join(patterns)
    return re.compile(rf"Yd pass from\s+(?:{name})", re.IGNORECASE)


def initiated(text: pd.Series, patterns: list[str]) -> pd.Series:
    """True where the quarterback initiated the play, in either text format."""
    clean = text.fillna("")
    return clean.str.contains(_actor_regex(patterns), regex=True) | clean.str.contains(
        _reverse_regex(patterns), regex=True
    )


def _mention_regex(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


def classify(play_type: str, text: str) -> str:
    """Bucket a play into dropback components or a designed run."""
    if play_type in SACK_TYPES or re.search(r"\bsacked\b", text, re.IGNORECASE):
        return "sack"
    if play_type in PASS_TYPES:
        return "pass"
    if play_type in RUSH_TYPES:
        return "rush"
    # Fall back to the text when play_type is a container like "Fumble Recovery".
    if re.search(r"\bpass(es)?\b", text, re.IGNORECASE):
        return "pass"
    if re.search(r"\b(rush|run|runs|scramble)", text, re.IGNORECASE):
        return "rush"
    return "other"


def extract(frame: pd.DataFrame, quarterback: str) -> pd.DataFrame:
    """Return every play ``quarterback`` initiated, with an attribution audit.

    The returned frame carries a ``play_kind`` column (pass / sack / rush) and
    keeps the raw ``play_text`` so any result can be traced back to the snap.
    """
    spec = QUARTERBACKS[quarterback]

    team_plays = frame[frame["pos_team"] == spec["team"]].copy()
    plays = team_plays[initiated(team_plays["play_text"], spec["patterns"])].copy()
    plays["quarterback"] = quarterback
    plays["play_kind"] = [
        classify(pt, tx) for pt, tx in zip(plays["play_type"], plays["play_text"].fillna(""))
    ]
    # A dropback is a pass attempt or a sack; designed runs are tracked separately.
    plays["is_dropback"] = plays["play_kind"].isin({"pass", "sack"})
    # A penalty can wipe a snap off the official books; such rows still carry EPA
    # for the penalty outcome, so they are flagged rather than dropped and each
    # analysis decides whether to count them.
    if "penalty_no_play" in plays:
        plays["nullified"] = plays["penalty_no_play"].fillna(0).astype(float) == 1
    else:
        plays["nullified"] = False

    # Pass depth and direction exist only in the verbose scorer format, so this
    # is populated for well under half of 2025 and must never be assumed present.
    depth = plays["play_text"].fillna("").str.extract(
        r"(?P<depth>short|deep)\s+(?P<direction>left|middle|right)", flags=re.IGNORECASE
    )
    plays["pass_depth"] = depth["depth"].str.lower()
    plays["pass_direction"] = depth["direction"].str.lower()
    # Scorer-credited hurries: a noisy lower bound on pressure, never a rate.
    plays["hurried"] = plays["play_text"].fillna("").str.contains("hurried", case=False)
    return plays


def audit(frame: pd.DataFrame, quarterback: str) -> dict:
    """Report how much of the team's passing game the parse actually captured.

    Attribution loss is the real risk in this dataset, so it is measured rather
    than assumed. ``team_dropbacks`` counts dropbacks by play_type alone, with no
    reference to any name; the coverage ratio against it exposes missed plays.
    """
    spec = QUARTERBACKS[quarterback]
    team = frame[frame["pos_team"] == spec["team"]]
    text = team["play_text"].fillna("")

    is_team_dropback = team["play_type"].isin(PASS_TYPES | SACK_TYPES)
    mention = _mention_regex(spec["patterns"])

    captured = initiated(team["play_text"], spec["patterns"])
    mentioned = text.str.contains(mention, regex=True)

    return {
        "quarterback": quarterback,
        "team_plays": int(len(team)),
        "team_dropbacks": int(is_team_dropback.sum()),
        "captured_plays": int(captured.sum()),
        "captured_dropbacks": int((captured & is_team_dropback).sum()),
        "dropback_coverage": float((captured & is_team_dropback).sum() / max(is_team_dropback.sum(), 1)),
        # Named but not in the actor slot: laterals, tackle credits, and the
        # penalty rows discussed below.
        "mentioned_not_actor": int((mentioned & ~captured).sum()),
        # Penalty rows carry no actor verb, so defensive pass interference —
        # a large positive, genuinely QB-generated outcome — is invisible here.
        "team_penalty_rows": int((team["play_type"] == "Penalty").sum()),
    }
