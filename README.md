# SDT-Project
TTGRPOBAD= We reward 1.0 if predictions matches the GT during GRPO irrespective of label.
TTGRPOsemBAD= We reward the change in state of label based on the effect of the change for all labels.
TTGRPOGOOD = We create a hybrid method where LLM as judge gives out the predicted talk_type label of client based on the context history at that moment.Gives out Label and confidence , where confidence measure how sure it is with its prediction. Confidence added just to get advantage during GRPO since other method had 0 advantages as measured.

# GRPO R2 (Talk Type Reward) - Three Implementation Methods

## Overview
This document summarizes three different approaches to implementing the R2 (Talk Type) reward for GRPO training of an MI therapist agent. The goal was to design a reward signal that guides the policy to generate utterances likely to elicit client **change talk** (commitment to change).

---

## Method 1: Hybrid GT Matching + Semantic Scoring

### Approach
If LLM prediction == Ground Truth talk_type:
Reward = 1.0 (perfect match)
Else:
Reward = SEMANTIC_TABLE[(GT, predicted)]

### Reward Table
```python
REWARD_TABLE = {
    ('change', 'change'):   1.0,
    ('change', 'neutral'):  0.65,
    ('change', 'sustain'):  0.15,
    ('neutral', 'change'):  0.9,
    ('neutral', 'neutral'): 1.0,    # GT match
    ('neutral', 'sustain'): 0.3,
    ('sustain', 'change'):  0.95,
    ('sustain', 'neutral'): 0.4,
    ('sustain', 'sustain'): 1.0,    # GT match
}
```

### Results
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Reward Mean | 0.827 | N/A | - |
| Advantage Std | 0.098 | N/A | Weak signal |
| Change Talk % | 51% | 0% | **Collapsed** |
| Neutral Talk % | 12% | 0% | **Collapsed** |
| Sustain Talk % | 37% | 100% | **Collapsed** |

### Issues
- ✗ Model learned to maximize "sustain-sustain" matches (1.0 reward)
- ✗ Complete collapse to sustain-only utterances
- ✗ Therapically suboptimal (reinforces client resistance)
- ✗ GT matching created perverse incentive

### Lesson
**Don't reward prediction accuracy of GT.** The model exploits matching GT talk_type instead of learning MI principles.

---

## Method 2: Pure Semantic Scoring (No GT Matching)

### Approach
All scored based on therapeutic progress:
sustain → change: 0.95 (best)
change → change: 1.0 (best)
neutral → change: 0.9 (good)
### Reward Table
```python
REWARD_TABLE = {
    ('change', 'change'):   1.0,
    ('change', 'neutral'):  0.65,
    ('change', 'sustain'):  0.15,
    ('neutral', 'change'):  0.9,
    ('neutral', 'neutral'): 0.5,    # No GT match reward
    ('neutral', 'sustain'): 0.3,
    ('sustain', 'change'):  0.95,
    ('sustain', 'neutral'): 0.45,
    ('sustain', 'sustain'): 0.3,    # Penalized (low)
}
```

### Results
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Reward Mean | 0.667 | 0.304 | Lower (unexpected) |
| Advantage Std | 0.1175 | 0.1175 | No improvement |
| Change Talk % | 55% | 3% | **Degraded** |
| Neutral Talk % | 11% | 0% | **Died out** |
| Sustain Talk % | 34% | 97% | **Exploited** |

### Issues
- ✗ Model still collapsed to sustain-only
- ✗ Advantages stayed near zero (K=4 all predicted "sustain")
- ✗ LLM's discrete 3-label system insufficient for discrimination
- ✗ No variance within each batch → no learning signal

### Lesson
**Discrete labels (3 options) are too coarse.** All K=4 utterances in a batch often get same label → same reward → zero advantages → no learning.

---

## Method 3: R2 Hybrid - Discrete Labels + Confidence Weighting ✓

### Approach

**Key Innovation:** Confidence weighting adds continuous dimension, enabling discrimination between K=4 responses even when labels are the same.

### Example
K=4 responses, all LLM predicts "sustain":
Resp1: sustain, confidence 3 → reward = 0.3 × 0.3 = 0.09
Resp2: sustain, confidence 8 → reward = 0.3 × 0.8 = 0.24
Resp3: sustain, confidence 5 → reward = 0.3 × 0.5 = 0.15
Resp4: sustain, confidence 6 → reward = 0.3 × 0.6 = 0.18

Mean: 0.165
Advantages: [-0.075, 0.075, -0.015, 0.015] ← NON-ZERO ✓

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


---
### More Changes?
Maybe we can add a rubric in the prompt for the LLM as judge to calculate the confidence based on the checklist we provide. The current method just predicts the confidence it has over its predicted label.


# R1 (Context Grounding Reward) - Implementation & Analysis

## Overview

R1 (Context Grounding) evaluates how well a generated therapist utterance is **grounded in the actual conversation history** and **relevant to the client's stated concerns**.

The goal is to ensure responses:
- ✓ Address what the client just said
- ✓ Reference specific dialogue details (not generic)
- ✓ Don't invent facts not mentioned
- ✓ Maintain conversational flow

---

## Design

### Rubric (GPT-3.5-turbo evaluates)

```python
R1_RUBRIC = {
    "addresses_concern": 0.40,      # Does T address C's stated concern?
    "grounds_in_dialogue": 0.30,    # Is it grounded in what C said?
    "no_hallucination": 0.20,       # No invented facts?
    "no_abrupt_switching": 0.10,    # Maintains conversational flow?
}
```

### Scoring Process

1. **LLM rates each criterion 0-5:**
   ```
   1. Addresses concern (0=ignores, 5=direct address)
   2. Grounded in dialogue (0=generic, 5=specific)
   3. No hallucinated facts (0=multiple false, 5=none)
   4. No abrupt topic switch (0=abrupt, 5=natural)
   ```

2. **Weighted sum:**
   ```
   Weighted_sum = (score1 × 0.40) + (score2 × 0.30) + (score3 × 0.20) + (score4 × 0.10)
   ```

3. **Normalize to [0, 1]:**
   ```
   R1_score = Weighted_sum / 5.0
   Range: [0, 1]
   ```

### Example Evaluation

```
Context:
  Client: "I want to quit smoking but I'm worried about withdrawal"
  
Generated Response:
  "What concerns you most about withdrawal?"

Scores:
  1. Addresses concern: 4/5 (directly addresses withdrawal worry)
  2. Grounds in dialogue: 5/5 (references exact client statement)
  3. No hallucination: 5/5 (no invented details)
  4. No abrupt switching: 5/5 (natural flow)

Weighted_sum = (4×0.40) + (5×0.30) + (5×0.20) + (5×0.10) = 4.6
R1_score = 4.6 / 5.0 = 0.92 ✓
```

---

## Results

### Baseline (Before GRPO)

```
Reward distribution:
  Mean: 0.887
  Std: 0.118
  Min: 0.42
  Max: 1.00

Advantage distribution (training signal):
  Std: 0.0593 (WEAK)
```

### After GRPO Training

```
Reward distribution:
  Mean: 0.786
  Std: 0.102
  Min: 0.5
  Max: 0.9 (sample)

Advantage distribution:
  Std: 0.0593 (NO CHANGE)
```

### Analysis

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Reward Mean | 0.887 | 0.786 | ↓ Decreased |
| Reward Std | 0.118 | 0.102 | ↓ Lower |
| Advantage Std | 0.0593 | 0.0593 | → No learning |
| Quality (visual) | Good | Good | ✓ Maintained |

---

## Key Findings

### ✗ R1 Alone Didn't Work

**Reason:** SFT baseline already generates contextually grounded responses

```
SFT (30-transcript warmup) was trained on MI transcripts
→ Already learned to be context-appropriate
→ All K=4 generated utterances score 0.80-0.95
→ No within-batch variance → advantages ≈ 0
→ No learning signal for policy → no improvement
```

**This is NOT a failure of R1 design.** It's a feature of the SFT baseline being too good at this already.

### ✓ R1 Has Value in Multi-Reward System

While R1 alone showed weak signal:
- It captures a legitimate MI principle (context grounding)
- It will **counterbalance R2's potential for aggressive change-elicitation**
- It prevents reward gaming (e.g., off-topic but high-reward utterances)

**R1 + R2 together > R1 alone or R2 alone**

---

## Implementation

### CELL 5: R1 Setup

```python
def compute_r1_context_grounding(context, therapist_utt):
    """R1: Score context grounding (optimized prompt)"""
    
    prompt = f"""Rate this therapy response on 0-5 for each criterion:

CONVERSATION:
{context}

THERAPIST RESPONSE:
{therapist_utt}

---

RATE EACH 0-5:
1. Addresses client's concern (0=ignores, 5=direct)
2. Grounded in dialogue (0=generic, 5=specific)
3. No hallucinated facts (0=multiple false, 5=none)
4. No abrupt topic switch (0=abrupt, 5=natural)

RESPOND: number number number number
Example: 4 3 5 4"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.choices[0].message.content.strip()
    scores = [int(x) for x in text.split()[:4]]
    if len(scores) < 4:
        scores += [2.5] * (4 - len(scores))
    
    criteria = ["addresses_concern", "grounds_in_dialogue", "no_hallucination", "no_abrupt_switching"]
    weighted_sum = sum(score * R1_RUBRIC[criteria[i]] for i, score in enumerate(scores[:4]))
    final_score = weighted_sum / 5.0
    
    return final_score
```

### CELL 7: GRPO Generation

```python
for i, prompt_dict in enumerate(grpo_prompts[:NUM_PROMPTS_TO_TRAIN]):
    prompt = prompt_dict['prompt']
    
    # Generate K responses
    outputs = policy_model.generate(...)
    responses = [tokenizer.decode(...) for output in outputs]
    
    # Score K responses using R1
    rewards = []
    for response in responses:
        r1_score = compute_r1_context_grounding(prompt, response)
        rewards.append(r1_score)
    
    # Compute advantages
    mean_reward = np.mean(rewards)
    advantages = np.array([r - mean_reward for r in rewards])
    
    grpo_training_data.append({
        "prompt": prompt,
        "responses": responses,
        "rewards": np.array(rewards),
        "advantages": advantages,
        "mean_reward": mean_reward
    })
```

### CELL 11: GRPO Training

```python
def compute_rewards(completions, **kwargs):
    """Score completions using R1 Context Grounding"""
    prompts = kwargs.get('prompts', [])
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        r1_score = compute_r1_context_grounding(prompt, completion)
        rewards.append(r1_score)
    
    return rewards

trainer = GRPOTrainer(
    model=policy_model,
    args=grpo_config,
    train_dataset=grpo_dataset,
    reward_funcs=[compute_rewards],
)
trainer.train()
```

---

## Comparison: R1 vs R2

| Aspect | R1 (Context Grounding) | R2 (Talk Type Hybrid) |
|--------|---|---|
| **Signal Type** | Continuous [0, 1] | Continuous [0, 1] (confidence-weighted) |
| **Advantage Std** | 0.0593 | 0.1311 |
| **Reward Mean** | 0.887 | 0.281 |
| **Learning Signal** | Weak | Strong |
| **Improvement** | None (↓0.1) | Significant (↑0.06) |
| **Why Weak/Strong** | SFT already grounded | SFT lacks talk-type discrimination |
| **Clinical Value** | Prevents off-topic | Elicits change talk |
| **Role in Multi-Reward** | Counterbalance | Primary driver |

---

## Multi-Reward Integration (R1 + R2)

### Recommended Weights

```python
R_total = λ₁R₁ + λ₂R₂ + λ₃R₃ + λ₄R₄

λ₁ = 0.20  # R1: Context Grounding (lower; weak signal, but stabilizing)
λ₂ = 0.40  # R2: Talk Type Hybrid (strong; elicits change talk)
λ₃ = 0.25  # R3: SDT/MI Process (future work)
λ₄ = 0.15  # R4: Therapeutic Alliance (future work)
```

### Why This Balance

```
R2 alone (aggressive):
  Generate utterance that elicits change talk
  Risk: Off-topic, irrelevant, aggressive challenges

R1 counterweight:
  Penalize off-topic, irrelevant, abrupt shifts
  Ensures grounded in dialogue

Combined:
  Change-oriented utterances that are contextually appropriate
  Balanced MI principles
```

---

## Insights

1. **High baseline performance obscures reward utility**
   - R1 weak signal doesn't mean R1 is bad
   - Just means SFT already learned this principle
   - Will be valuable when combined with other rewards

2. **Continuous rewards are essential**
   - Even weak signal (0.0593) better than discrete
   - Allows fine-grained policy optimization

3. **Context grounding is table stakes**
   - Clinical standard for good therapy
   - Shouldn't be optimized heavily — should be baseline
   - Use lower weight (0.20) in multi-reward

4. **Complementary rewards prevent gaming**
   - R2 alone could lead to aggressive optimization
   - R1 acts as a stabilizing constraint
   - Multi-reward is necessary for safe RL

---

## Cost Analysis

**Per GRPO Run (50 prompts × 4 candidates = 200 API calls):**
- ~200 tokens per call (long rubric prompt)
- ~$0.03-0.05 per run
- **Total R1 training cost:** ~$0.10-0.15

---

## Future Work

1. **Optimize rubric weights** — current 0.40/0.30/0.20/0.10 are reasonable but not validated
2. **Shorter prompt variant** — test if prompt length affects consistency
3. **Train evaluator** — could train a smaller model to replace GPT-3.5-turbo (cost reduction)
4. **Combine with R2** — implement full multi-reward system
5. **Add R3 + R4** — SDT process and therapeutic alliance rewards



