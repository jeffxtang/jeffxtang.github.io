---
layout: post
comments: true
title:  "Seemingly Small Changes with Big Impact in Modern AI"
date:   2025-10-04 09:18:00
categories: ai, reasoning, rl
---

Having been living with GenAI and thinking about the transformer and MoE, which BTW was amazingly origined in this 1991 paper [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf), the year another [dinasour](https://www.espn.com/blog/new-york/brooklyn-nets/post/_/id/17845/old-man-pierce-delivers-playoff-daggers) got his master's degree in classical AI, I suddenly had the urge this Saturday morning to ask AI this question: 

"give me examples in the deep learning and genai field where one seemingly small change or adoption of a method like in activation function or dropout or the use of sqrt of embedding dimension in the attention formula significantly changed the trained model performance"

then with two follow ups:

"does change from relu to gelu also have a big impact? also give me as many additional real examples as possible."

"how does one come up with such seemingly small but very impactful changes? lots of systematic experiments and/or research or engineering intuition or deep thinking and reasoning?"

Exciting answers from Google AI Mode are [here](https://share.google/aimode/ELIMWICiz1JEeIgvx), with the summary to the answer to the last question as follows:

In practice, the process is a virtuous cycle:

1. Deep thinking identifies a theoretical problem or limitation.

2. Intuition proposes a candidate solution.

3. Systematic experimentation validates the idea and refines its implementation.

4. The results from these experiments build deeper intuition and inform the next round of ideas. 

That "intuition" reminds me of what's written in a nice book I read recently - How to Beat Your Dad at Chess by Murray Chandler, albeit likely inappropriately titled because it's readlly not just for kids:

"Inexperienced players use perhaps 95% calculation and 5% pattern recognition. For master-strength players the figure is more like 40% calculation and 60% pattern recognition. Logically, therefore, learning to recognize more key patterns could help dramatically improve your chess strength."
