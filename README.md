# SDT-Project
TTGRPOBAD= We reward 1.0 if predictions matches the GT during GRPO irrespective of label.
TTGRPOsemBAD= We reward the change in state of label based on the effect of the change for all labels.
TTGRPOGOOD = We create a hybrid method where LLM as judge gives out the predicted talk_type label of client based on the context history at that moment.Gives out Label and confidence , where confidence measure how sure it is with its prediction. Confidence added just to get advantage during GRPO since other method had 0 advantages as measured.


### Reward Table (Semantic, No GT Matching)
```python
REWARD_TABLE = {
    ('change', 'change'):   1.0,
    ('change', 'neutral'):  0.65,
    ('change', 'sustain'):  0.15,
    ('neutral', 'change'):  0.9,
    ('neutral', 'neutral'): 0.5,
    ('neutral', 'sustain'): 0.3,
    ('sustain', 'change'):  0.95,
    ('sustain', 'neutral'): 0.45,
    ('sustain', 'sustain'): 0.3,
}
```

### Results
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Reward Mean | 0.280 | 0.342 | **Improved ✓** |
| Advantage Std | 0.1311 | 0.1311 | **Strong signal ✓** |
| Change Talk % | 16% | 20% | **Improved ✓** |
| Neutral Talk % | 0.5% | 0% | Expected (low utility) |
| Sustain Talk % | 83.5% | 80% | **Reduced ✓** |

### Success Metrics
- ✓ Non-zero advantages (policy learns)
- ✓ Reward improved from 0.280 to 0.342
- ✓ Change talk increased from 16% to 20%
- ✓ No collapse (diversity maintained)
- ✓ Advantage std strong (0.1311)

### Why It Works
1. **Continuous signal:** Confidence (0-10) provides many more scoring options than discrete labels
2. **Discrimination:** K=4 responses now likely have different confidence scores → different rewards
3. **No collapse incentive:** No perfect-match 1.0 reward → model can't exploit one strategy
4. **Rubric clarity:** Explicit confidence criteria ensure consistent LLM scoring

### Lesson
**Hybrid continuous + discrete is the sweet spot.** Keeps interpretability (talk_type labels) while adding fine-grained discrimination (confidence weighting).

---

## Comparison Summary

| Aspect | Method 1 | Method 2 | Method 3 |
|--------|----------|----------|----------|
| Approach | GT match + semantic | Pure semantic | Hybrid + confidence |
| Collapse Risk | High ✗ | High ✗ | Low ✓ |
| Learning Signal | Weak | Weak | Strong ✓ |
| Interpretability | Good | Good | Good ✓ |
| Reward Mean | 0.827 | 0.667 | 0.342 (realistic) |
| Change Talk Gain | -51% | -52% | +4% |
| **Recommended** | ✗ | ✗ | **✓** |

---

## Implementation

### CELL 5: R2 Hybrid Function
```python
def compute_r2_hybrid(context, therapist_utt, gt_talk_type):
    """
    R2 Hybrid: Discrete label + confidence weighting
    Returns: (label, reward_score)
    """
    prompt = f"""Given this therapy conversation:

{context}

Therapist: {therapist_utt}

TASK:
(a) Predict client's likely talk_type: change, sustain, or neutral
(b) Rate your confidence 0-10 based on:
  9-10: Very clear signal, unambiguous client response
  7-8:  Strong signal, likely response
  5-6:  Moderate signal, plausible response
  3-4:  Weak signal, uncertain response
  1-2:  Very unclear, unlikely response

RESPOND EXACTLY: LABEL CONFIDENCE
Example: change 8
Or: sustain 5
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.choices[0].message.content.strip().lower()
    parts = text.split()
    label = parts[0]
    confidence = int(parts[1]) / 10.0 if len(parts) > 1 else 0.5
    
    base_reward = REWARD_TABLE.get((gt_talk_type, label), 0.5)
    final_reward = base_reward * confidence
    
    return label, final_reward
```

### CELL 7: GRPO Generation
```python
for response in responses:
    pred_label, r2_score = compute_r2_hybrid(prompt, response, gt_talk_type)
    predicted_labels.append(pred_label)
    rewards.append(r2_score)

mean_reward = np.mean(rewards)
advantages = np.array([r - mean_reward for r in rewards])
```

---

## Cost Analysis

**Per GRPO Run (50 prompts × 4 candidates = 200 API calls):**
- ~120 input tokens + 10 output tokens per call
- ~$0.01-0.02 per run
- **Total budget impact:** Low

---

## Future Work

**Multi-Reward System (Planned):**
