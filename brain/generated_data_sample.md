# Route-Format Training Data Sample

Example record from `core/synthetic_data/generators/temporal/generator.js`:

```json
{
  "text": "My medication alarm goes off at 09:58. It takes 5 minutes to get to the kitchen. What time will I arrive?\n[ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_58] [HEAD_DURATION] [ARG_MIN_05] [/ROUTE]"
}
```

**Key points:**
- No `[ANSWER]` block — model only learns to produce `[ROUTE]` tokens
- The Luxon engine computes `09:58 + 5min = 10:03` deterministically
- Shadow pairs teach base-10 vs base-60 distinction:

```json
{
  "text": "What is 58 + 5?\n[NO_ROUTE] This is base-10 arithmetic, not temporal."
}
```

- Negative examples prevent over-triggering on non-temporal numbers:

```json
{
  "text": "I scored 1500 points in the game today. That's 500 more than yesterday.\n[NO_ROUTE] Not a temporal expression."
}
```
