---
layout: post
comments: true
title:  "Reinforcement Learning in Tic-Tac-Toe"
date:   2017-01-06 09:52:06
categories: reinforcement learning, Swift, iOS, AI
---

Different people may learn in different ways. Some prefer to have a teacher, a mentor, a supervisor, guiding them on each step in their learning process; others are more like self-learners. Supervised learning no doubt is very important, but I found reinforcement learning (RL) fascinating because of its self learning flavor - a program called TD-Gammon developed in 1992 learned how to play backgammon via hundreds of thousands of self-play games and was then able to beat the best human player. The famous AlphaGo also used RL - see [Jim Fleming's Before AlphaGo there was TD-Gammon](https://medium.com/jim-fleming/before-alphago-there-was-td-gammon-13deff866197#.q4gqxgi47) for more info.

The idea of playing thousands of, or hundreds of thousands of, or even millions of games against oneself and then becoming the best player in the world sounds too appealing to not get my hands dirty. So in October I started implementing algorithms in the classic book by Richard Sutton [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/%7Esutton/book/the-book-1st.html) (BTW, a former student of Sutton, David Silver, is the tech lead of AlphaGo). My first implementation, naturally, was the extended example Tic-Tac-Toe described in the first Chapter of the book, where the important temporal difference learning algorithm is covered. I also decided to use Swift as the language, so I won't get too rusty on it with my obsession of the AI world most of the time.

After I got the TD(0) algorithm implemented in Swift and witnessed some interesting and satisfactory testing results, I just moved on, trying to implement the TD(lambda) algorithm in TensorFlow, which I'll talk about in a future blog, and to build an iOS app that'd play a fun game called [Quarto](https://en.wikipedia.org/wiki/Quarto_(board_game)) and beat everyone because it'd become the best player after learning from playing lots of games against itself, just like TD-Gammon. I didn't think writing about my timid start with RL with Tic-Tac-Toe would be worthwhile.

Then in the last day of 2016, while I was bathing in the warm winter California sunshine, reading von Goethe's The Sorrows of Young Werther, a book I read for the first time in 1980's in between the 4-volume Handbook of Artificial Intelligence when the second [AI winter](https://en.wikipedia.org/wiki/AI_winter) was about to begin, what then-24-year-old Goethe wrote in that October of 1771 really stroke me: "When, in spite of weakness and disappointments, we set to work in earnest, and persevere steadily, we often find, that, though obliged continually to tack, we make more way than others who have the assistance of wind and tide." It just feels extremely sensible for me to write about my Swift implementation of RL algorithm on Tic-Tac-Toe in another October, 245 years later.

Or I could just recall the old Chinese saying "A journey of a thousand miles begins with a single step", where the word "step" looks so fitting (hopefully not overfitting) in the training world of machine learning, to justify writing this RL Tic-Tac-Toe blog.

Enough talking. So the code is in [the RLTicTacToe repo](https://github.com/jeffxtang/RLTicTacToe). I basically implemented the TD(0) learning method `V(s) = V(s) + alpha * (V(s') - V(s))`, where alpha is the learning rate, s is a game state before the greedy move (called the best move in my code), s' is the state after the move, and V is the estimated value of a state. Initially all game states have values of 1.0 (the player making the first move, X, wins), 0.0 (the opponent player, O, wins), or 0.5 (draw).

Some notes on the code in main.swift:

* Unique non-terminal states for the Tic-Tac-Toe game are generated, which could take 3-5 minutes, and saved using NSUserDefaults and restored from the second time the app runs. This way learning and testing after the first time can be much faster.

* The learned state values are not currently saved between app runs. This can be handy when comparing learning performance during app testing, but saving the learned state values (or, more generally, the learned model) would be useful when the app plays against a human user. In a future post, I'll show you how a learned model trained with TensorFlow and Python can be used directly in an iOS app.

* The `evaluate_self_play_mode` function is used to compare the results when choosing the greedy (best) or random move, with or without learning. If the best move is selected, then in the function `select_next_best_move`, the first player chooses, among all possible next states, the state with the maximum value so it can get as close to winning (value 1.0) as possible, while the second player chooses the minimum value to be close to its winning state (value 0.0).

* The recursive `self_play` function is used for each complete game episode. If the learning is enabled, the TD(0) update rule described above is implemented in `unique_states[index].value = s2.value + alpha * (s1.value - s2.value)`. Notice that after each move by a player, the value of the state resulted from the previous move by the SAME player gets updated.

* With 4 different combinations of random/best move and with/without learning, we get the following results:

```
after random move without learning 100, win: 56, loss: 30, draw: 14
after random move without learning 200, win: 57, loss: 29, draw: 14
after random move without learning 300, win: 52, loss: 35, draw: 13
after random move without learning 400, win: 60, loss: 29, draw: 11
after random move without learning 500, win: 63, loss: 31, draw: 6
after random move without learning 600, win: 61, loss: 25, draw: 14
after random move without learning 700, win: 62, loss: 27, draw: 11
after random move without learning 800, win: 60, loss: 32, draw: 8
after random move without learning 900, win: 56, loss: 31, draw: 13
after random move without learning 1000, win: 52, loss: 32, draw: 16

after best move without learning 100, win: 67, loss: 29, draw: 4
after best move without learning 200, win: 74, loss: 25, draw: 1
after best move without learning 300, win: 64, loss: 30, draw: 6
after best move without learning 400, win: 79, loss: 18, draw: 3
after best move without learning 500, win: 70, loss: 28, draw: 2
after best move without learning 600, win: 64, loss: 31, draw: 5
after best move without learning 700, win: 65, loss: 30, draw: 5
after best move without learning 800, win: 65, loss: 27, draw: 8
after best move without learning 900, win: 72, loss: 24, draw: 4
after best move without learning 1000, win: 63, loss: 31, draw: 6

after random move with learning 100, win: 58, loss: 28, draw: 14
after random move with learning 200, win: 60, loss: 28, draw: 12
after random move with learning 300, win: 61, loss: 24, draw: 15
after random move with learning 400, win: 56, loss: 33, draw: 11
after random move with learning 500, win: 60, loss: 33, draw: 7
after random move with learning 600, win: 64, loss: 25, draw: 11
after random move with learning 700, win: 63, loss: 29, draw: 8
after random move with learning 800, win: 59, loss: 28, draw: 13
after random move with learning 900, win: 57, loss: 34, draw: 9
after random move with learning 1000, win: 62, loss: 32, draw: 6

after best move with learning 100, win: 80, loss: 18, draw: 2
after best move with learning 200, win: 96, loss: 3, draw: 1
after best move with learning 300, win: 94, loss: 3, draw: 3
after best move with learning 400, win: 91, loss: 9, draw: 0
after best move with learning 500, win: 91, loss: 9, draw: 0
after best move with learning 600, win: 95, loss: 2, draw: 3
after best move with learning 700, win: 93, loss: 5, draw: 2
after best move with learning 800, win: 96, loss: 3, draw: 1
after best move with learning 900, win: 99, loss: 1, draw: 0
after best move with learning 1000, win: 95, loss: 4, draw: 1
```

This is exactly what we expected: selecting best moves (without learning) should result in more wins for the first player than selecting random moves, and selecting best moves with learning during self play should be the best.

* Other features of RL, such as making exploratory moves, using eligibility trace in the TD(lambda) algorithm, and saving the trained model, have not been covered in the project and this post yet. But I'll soon talk about these in a TensorFlow based example project.

This toy implementation of RL is indeed a very small step towards the ambitious goal of AI. But I hope that people newly interested in RL will see from this example that self play can indeed learn and works, and that those who prefer to see the actual running code, ideally comfortable with the iOS and Swift world, will have a little treat to enjoy and build future work on.

Or this is just one proof I can refer to in the future that "in spite of weakness and disappointments"...
