import numpy as np
import pandas as pd

nAgent = 4
short = 15
nround = 8
gridSize = 121
side = int(np.sqrt(gridSize))
Xstar = np.array([(x, y) for x in range(side) for y in range(side)])

if __name__ == "__main__":
    data = pd.read_csv("./data/e1_data.csv")

    # Clean reward (participants never saw rewards lower than .05)
    data.loc[data["reward"] < 0.05, "reward"] = 0.05

    # Add all columns for behavioral measures
    new_cols = [
        "last_choice_distance_private", "nearest_choice_distance_private", "avg_choice_distance_private",
        "last_choice_distance_social", "nearest_choice_distance_social", "avg_choice_distance_social",
    ]
    for col in new_cols:
        data[col] = np.nan

    for g in np.unique(data["group"]):
        for r in np.unique(data["round"]):
            for ag in np.unique(data.loc[data["group"] == g, "agent"]).astype(int):
                agent_trials = data.loc[
                    (data["agent"] == ag) & (data["group"] == g) & (data["round"] == r)
                ].sort_values("trial")
                trials = agent_trials["trial"].values
                choices = Xstar[agent_trials["choice"].astype(int).values]
                rewards = agent_trials["reward"].values

                for idx, t in enumerate(trials):
                    if idx == 0:
                        continue

                    mask = (
                        (data["agent"] == ag)
                        & (data["group"] == g)
                        & (data["round"] == r)
                        & (data["trial"] == t)
                    )

                    prev_choice = choices[idx - 1]
                    curr_choice = choices[idx]

                    # private
                    data.loc[mask, "last_choice_distance_private"] = np.linalg.norm(
                        curr_choice - prev_choice
                    )
                    prev_choices = choices[:idx]
                    dists = np.linalg.norm(curr_choice - prev_choices, axis=1)
                    data.loc[mask, "nearest_choice_distance_private"] = dists.min()
                    data.loc[mask, "avg_choice_distance_private"] = dists.mean()

                    # social
                    social_choices = Xstar[
                        data.loc[
                            (data["agent"] != ag)
                            & (data["group"] == g)
                            & (data["round"] == r)
                            & (data["trial"] <= t),
                            "choice",
                        ]
                        .astype(int)
                        .values
                    ]
                    if len(social_choices) > 0:
                        dists_soc = np.linalg.norm(curr_choice - social_choices, axis=1)
                        data.loc[mask, "nearest_choice_distance_social"] = dists_soc.min()
                        data.loc[mask, "avg_choice_distance_social"] = dists_soc.mean()

                    last_social_choices = Xstar[
                        data.loc[
                            (data["agent"] != ag)
                            & (data["group"] == g)
                            & (data["round"] == r)
                            & (data["trial"] == t - 1),
                            "choice",
                        ]
                        .astype(int)
                        .values
                    ]
                    if len(last_social_choices) > 0:
                        dists_last_soc = np.linalg.norm(
                            curr_choice - last_social_choices, axis=1
                        )
                        data.loc[mask, "last_choice_distance_social"] = (
                            dists_last_soc.mean()
                        )

    data.to_csv("data/e1_data_extended.csv", index=False)
