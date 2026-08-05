# SDT-Project
TTGRPOBAD= We reward 1.0 if predictions matches the GT during GRPO irrespective of label.
TTGRPOsemBAD= We reward the change in state of label based on the effect of the change for all labels.
TTGRPOGOOD = We create a hybrid method where LLM as judge gives out the predicted talk_type label of client based on the context history at that moment.Gives out Label and confidence , where confidence measure how sure it is with its prediction. Confidence added just to get advantage during GRPO since other method had 0 advantages as measured.
