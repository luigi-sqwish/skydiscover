"""Split-aware prompt evaluator used as a reference benchmark."""


def evaluate(program_path, *, split="train", phase="search"):
    with open(program_path, "r", encoding="utf-8") as f:
        prompt = f.read().lower()

    score = 0.1
    feedback = []

    if "carefully" in prompt:
        score += 0.2
        feedback.append("Preserves careful reasoning guidance.")

    if split == "train":
        if "directly" in prompt:
            score += 0.5
            feedback.append("Good train performance: concise answers help the train split.")
        else:
            feedback.append("Train split prefers concise prompts.")
    elif split == "val":
        if "verify" in prompt or "double-check" in prompt:
            score += 0.6
            feedback.append("Good validation performance: explicit verification generalizes better.")
        else:
            feedback.append("Validation split prefers prompts with explicit verification.")

    if phase == "final":
        score += 0.05
        feedback.append("Final phase uses the authoritative score.")

    return {
        "combined_score": round(score, 4),
        "artifacts": {
            "feedback": " ".join(feedback),
        },
    }
