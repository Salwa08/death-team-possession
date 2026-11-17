import numpy as np
FPS = 25
# Conservative physical sanity limit, not a learned threshold.
MAX_PLAUSIBLE_PLAYER_SPEED = 15.0  # m/s

def extract_evidence(
    players_prev, players_now,
    ball_prev, ball_now
):
    """
    All positions are in metres.

    Players: arrays of shape (46, 2).
    Slots 0:24 = H, 24:46 = A.

    Returns measurements, NOT possession labels.
    """
    previous = np.asarray(players_prev, dtype=float)
    current = np.asarray(players_now, dtype=float)
    b_prev = np.asarray(ball_prev, dtype=float)
    b_now = np.asarray(ball_now, dtype=float)

    result = {}

    for team, slots in (
        ("H", range(0, 24)),
        ("A", range(24, 46)),
    ):
        candidates = []

        if np.isfinite(b_now).all():
            for slot in slots:
                if np.isfinite(current[slot]).all():
                    distance = np.linalg.norm(
                        current[slot] - b_now
                    )
                    candidates.append((distance, slot))

        if not candidates:
            result[team] = {
                "distance": np.nan,
                "relative_speed": np.nan,
                "nearest_slot": -1,
            }
            continue

        distance, slot = min(candidates)

        relative_speed = np.nan

        if (
            np.isfinite(b_prev).all()
            and np.isfinite(previous[slot]).all()
        ):
            ball_velocity = (b_now - b_prev) * FPS
            player_velocity = (
                current[slot] - previous[slot]
            ) * FPS

            player_speed = np.linalg.norm(player_velocity)

            if player_speed <= MAX_PLAUSIBLE_PLAYER_SPEED:
                relative_speed = np.linalg.norm(
                    ball_velocity - player_velocity
                )

        result[team] = {
            "distance": float(distance),
            "relative_speed": float(relative_speed),
            "nearest_slot": int(slot),
        }

    h = result["H"]["distance"]
    a = result["A"]["distance"]
    

    result["distance_margin_H"] = (
        float(a - h)
        if np.isfinite(h) and np.isfinite(a)
        else np.nan
    )

    return result

def empty_players():
    return np.full((46, 2), np.nan)