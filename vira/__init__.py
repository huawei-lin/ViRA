"""ViRA: visual agents trained with Cross-Input Relative Policy Optimization (CrossRPO).

common    every number of the method (pixel budget, image sizes, the CrossRPO advantage, answer parsing)
protocol  prompts, tool calls, the zoom tool's boxes and crops, the reward
agent     one rollout of the zoom agent against a served model (evaluation and inference)
hf        models and data from the Hugging Face Hub
train     training on verl: agent loop, zoom tool, the CrossRPO advantage, trainer, entry point
eval      validation inside training and the offline evaluation
"""
