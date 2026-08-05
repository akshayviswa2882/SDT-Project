# SDT-Project
TTGRPOBAD= We reward 1.0 if predictions matches the GT during GRPO irrespective of label.
TTGRPOsemBAD= We reward the change in state of label based on the effect of the change for all labels.
TTGRPOGOOD = We create a hybrid method where LLM as judge gives out the predicted talk_type label of client based on the context history at that moment.Gives out Label and confidence , where confidence measure how sure it is with its prediction. Confidence added just to get advantage during GRPO since other method had 0 advantages as measured.



#LLM generated overview of the methods we did based on how I explained the information to it:

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
| **Recommended** | ✗ | ✗ | **✓** |

---



