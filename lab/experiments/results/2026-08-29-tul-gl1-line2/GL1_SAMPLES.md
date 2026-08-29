# Generation samples — GL1 line checkpoints (2026-08-29)

512 new tokens per sample, seed 1234, 12 prompts fixed across arms, max_slots 160. Rank on sampled decoding; greedy is a diagnostic (argmax-loop detector), not a quality metric. All four arms decode through the TUL slot machinery (slots regenerate as the model emits boundaries).

Val CE @4250 for these checkpoints: gl1b 4.4047 < gl1c 4.5974 < ctrl 4.6656 < gl1 5.0033 (A3 reference 4.02).

## Degeneration metrics (vs real-text anchor at the same length)

| arm | decode | rep4 | distinct3 |
|---|---|---|---|
| REAL TEXT (OWT held out, n=256) | — | 0.0365 | 0.9276 |
| gl1b | topk50_t0.8 | 0.0252 | 0.9175 |
| gl1b | sample_t1 | 0.0005 | 0.9974 |
| gl1b | greedy | 0.8160 | 0.1502 |
| gl1c | topk50_t0.8 | 0.0218 | 0.9190 |
| gl1c | sample_t1 | 0.0008 | 0.9954 |
| gl1c | greedy | 0.7785 | 0.1846 |
| gl1 | topk50_t0.8 | 0.0193 | 0.9364 |
| gl1 | sample_t1 | 0.0007 | 0.9969 |
| gl1 | greedy | 0.8836 | 0.0933 |
| ctrl | topk50_t0.8 | 0.0174 | 0.9397 |
| ctrl | sample_t1 | 0.0002 | 0.9975 |
| ctrl | greedy | 0.8879 | 0.0904 |

Read: every arm's topk50 rep4 sits UNDER the real-text anchor (0.017–0.025 vs 0.0365) with distinct3 within ~0.01 of real text — no degeneration at 512 tokens. Greedy loops on all arms as expected at this scale; the mask arms loop least under greedy (gl1c 0.779, gl1b 0.816 vs ctrl 0.888).


---

# GL1b — gist mask + MUX write target (line champion) (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0196, distinct3 0.9275, 22 spans, mean span 23.4)

> The theory of relativity states that some of the most expensive people would be to know about the real world today that they could be the same as the first thing to are.
> , of course, the more than the economic power that is in the economy, is the need to be a conferences.
> 
> At the same time, the economy is a very natural position and a lot of the Americans who are at a very good time.30.
> In the last three years, the 18th century is more than half the 11% of the population ever became the next war since the the world of the 1960s was the biggest issue in the history of the world.
> 
> But so on, economic and economic growth comes from the US, and the world has been the next economy.I am just not a young politician who has an academic and a national policy, and a lot of people are getting involved and a lot of economic.
> 
> The economy is a real problem because it is an important way, and the economic market is not the same as what’s really the to be doing.We have a plan, I am an economic strategy.But, to say, what we are talking about is the world’s best interest in the world.
> 
> “The government has more than a million, low interest rates now, and the United States is going to be a “comparable” in my country to a great deal of countries.So, as opposed to the world, the vast majority of income has been a positive effect to the EU.’ And now, it’s not about the interests of Americans to the government, and where it’s not as much in the world.“The world is going to become a positive interes…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0255, distinct3 0.9078, 23 spans, mean span 22.2)

> Once upon a time in a distant land, there lived a plea downside of a dorm in the 900s and that a I wondered why I was only a stomp of it.
> 
> The fact that you’ve had a few more people and the loans I’d been sure that the death of the breed was.I didn’t know how much I could.
> 
> On the other hand, this was a good idea to look at us at a time.-I guess I really want to try to believe it was in this piece of sense by the fact that it was better for me— who are in the girl, I think, you want to write.
> 
> A crowd of a couple of weeks, and I feel like I didn’t think that when you have a fetish baby, you really good.
> I don’t believe the same thing in the past year, but I think a lot of stuff I ever just wanted to do it.
> 
> “I’m not going to get a little bit of a punch.—I think I could do things so much for them.”
> 
> “What I’d have is a 20-year-old, you’re working on the same name.Well, I’m going to say myself,” she said.
> 
> It’s still hard to say that the whole thing is, it was the main reason why I’ve ever done anything: we will be a.
> 
> “If you know what they want, they are a fascism of this one day.“I know there’s a lot of time, and you don’t have to be a good one.
> 
> “I think that was a lot of person.Oh, I was a lot of people I’m hoping that it would not be a lot of work.”
> 
> “SI’ve only seen a lot of time with a woman, I’m proud of it.I’m a girl and a lot of people like me to say that I’d been making a fantasy year for all people and everything I’ to be a young fuck about it because you might not…

### top-k 50, T=0.8 — prompt 2  (rep4 0.0098, distinct3 0.9333, 26 spans, mean span 19.8)

> In machine learning, the key insight is that the team is able to take a to take the best way to change the way the world is on the side.14 and 64–.
> 
> “The best way to understand is that the game is being found in the world is, the best in the world that is the result of this is the world.Is it a crazy problem to get it?
> 
> The U.R.R., has said it is more like a “real” and “we’ve been a big fan,” and the team’s biggest experience in the world with.
> 
> “If we want to go to the right world, not the best game of this, but we’ve taken an opportunity to take out this a way that we’ve seen the biggest step of the game at around this level,” Bloomberg said.Everyone has a lot of players who thinks that the team will have as much support for this team, and he has already got his own experience with the to be a bit more aimed at getting away from the team.But we will also show more players around the world.But the most important question is that the team is a good man.
> 
> “We’re very afraid the player can do some stuff, but they can be a good thing for a couple of reasons.’
> 
> “I have no idea how to go and how well they’d be able to win,” said Pakistan..
> 
> Bruce Gates, who has been the biggest in the US and the AFL has taken a lot of time to have a team that gets into the season, in the middle of the season.The Dennis team still has 300 team team game game.And if the team is good, the player is a good team of the club and a young game.
> 
> GOP Williamson is moving onto the team around the game and h…


---

# GL1c — curriculum (warm-start from unmasked ctrl @2000) (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0020, distinct3 0.9765, 26 spans, mean span 19.8)

> The theory of relativity states that some of the most expensive people are willing to take a deal with a full deal.
> 
> This is not the first thing to do.
> 
> They are working on a real life, or it has been seen as a result of the society to be imposed by the fact that the rest, or the subject of the political process is the “wealth of the world,” a couple people were at a real man. Museum says that no laws of the conservative power is still in danger for the right to have a very simple job. I remember how people are going to use my own stories on how they can be.
> 
> I was a bad sense about how different this is it is to be the only one that was to accept and to make sure that the fascist of them who are really very bad about the world that I can’t do. I think, "When you were really sure, you have to do it because you should get out of the holiday of your hands. You can also be a lot of your friends.”
> 
> It didn’t happen here, but that is what we are. The same thing is that we have a right, I am not going to have to go back to our house with things I want to see. you need to do that when you’re playing the day where you can get back? We have a lot of reasons in these ways as well. I don’t think of the fact that my family is an example of what I was doing. I know that the more you are and they are, ‘You could be like to look at this, and they’re going to get your favorite and you’re an idea you can learn about. you can’t do it to your thoughts.” if it works with me, it might be the time…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0216, distinct3 0.9196, 27 spans, mean span 19.3)

> Once upon a time in a distant land, there lived a plea dramatic, a girl that was a pretty different dress with a Napolis. and I will be a much different way to make the fact that you’ve had a few more consequences of a cushstian, or a few the past three years.I think the fact that there are only a few times where the shit is a lady who gets a good idea about whether they were more good for all is going to meet. So we’ve been pretty much in this world. So it’s hard to do everything you’re going to look at.
> 
> The second question is that the first of the 90s had a little more than a little bit than the second time.5000 is a way to have a baby tends as a mouth, but some of them are not actually in the past year. But we’ll be very nice, especially the fact that the 17th and 13th-old is a little bit. I thought this was a few games I could do on my own, because I'm not really wrong, I don’t think they are going to be done, I think it’s great that I want to understand, more people. I think this one thing is this. I’m really like that, and I’m just able to see your own, but I don’t really really really kinda any other. I don’t think I’m not saying, that’s a good thing about being done on the day. Maybe I won’t be a hard way to move out of the way to try, but I’m just saying what you like. that was a lot of thing. I think it was a very good time I’m hoping that it has not seen as a big thing so it’s going to be pretty bad, so for that I have a very very good place. I think that’s very h…

### top-k 50, T=0.8 — prompt 2  (rep4 0.0452, distinct3 0.8843, 26 spans, mean span 19.8)

> In machine learning, the key insight is that 12% of the most interesting, 20 percent of the most part in the debt. the best of the 40-year-old ration, which is 13 times, is being found dead from the 46th cent-off with 1.00% of the world’s low-inch towns of U. the U.S.5 billion (100-202) in the next two years.74% of 162, while 25% of the 93 percent of their infrained 4168, the perception of 144.157 million per cent of the NFL members, and 455% of the total increase, this year and in 80 years  42th).
> 
> While the rest of the sequel is over 80,000,000,000 yards, while the 50 percent of the average tax dollars.347 of the 4,300-day countries, 2007, the only way to be considered in the 19 the years; and 100,000 Americans have to keep their nuclear rally from 5.05 percent of the new workers in federal China (2:22 percent).; 30 percent of those banks are already in the same year, with 27 percent of the current economic change. they are not able to have to defend the average risk, in order to increase the overall $2.25% per year, according to 0.9 percent and $4.illion, the rate of 200 in 2015.28 percent of the Americans are being able to pay for their children to their 15,000 factions to the 20 terrorist offices in the past 2007. A2008-03-60 of those fought a few weeks later, though the next time the 95 percent of the average economy is 1998.73 2017 60% 25% points
> 
> The university has now been overwhelmingly in the past 188 years.20% of the new population


---

# GL1 — gist mask, no MUX (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0118, distinct3 0.9412, 24 spans, mean span 20.8)

> The theory of relativity states that some of the most expensive can be taken to the same. the same is that these problems are the only way the first thing we are. we, our way to be able to ensure that the government has been in the same way of the new one to the next two years. they have an excellent amount of money and their own things, and we all know how to do a lot of people. do more than a few months. are in the same time, but they are now trying to stay in. there is a good way if there is a little bit. have a much more time to read this, so we see how they can be. in our same time, the the majority of the “Musalem,” the BBC campaign says. the crowd has always done the fascist case for the world. it is also the case that I can’t do so in those ways, because with the point they have all involved with them. that’s not a big deal to continue to say, but is a very low one. only can we see a lot, such a new way that is not the same time I’d see the same way if it’s not to get down in our world, you’ll see an extra thing that has a huge effect of the country to get us to have to be the only way to get them for more than I believe, ‘In the UK, I’m going to do it all that way of the fact I do like, but I had to say I’ve to know that the more people are going to be a ‘to me, all to look like he’s now, and I’m’d going to be. do you don’t love some of it, and I’m in the beginning of the story, “When I’ll know the thing is. that’s an important way for things you can do.”
> 
> However, o…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0118, distinct3 0.9667, 21 spans, mean span 24.8)

> Once upon a time in a distant land, there lived a plea to the most important dronics that have been dispronesated with a part of the horse and flick on the next few years. all the fact that you’ve been a few more than 100 years, including in the time, the company is actually the only. the fact that there are only four points in the world?
> 
> The idea that this is not the idea that the pooling has been all of them to take a major 10 per cent of the g worlds. This has been released by the United States as a result as the Missouri-Israel of the U.-0993s (1954-148/1950), an example of the Church’s second generation, as a new list of time-and-backed drune of GBA Church, a 11-year-old-year-old-taking-down in Germany, 1983. had been the first shocking in the 1960s in 1996, and the 1880K-1,000 Mexican murder, is a strong and unprecedented, in this project. to 1999, 1978, the 1990s, then included an increase by the “a 45 and 2010,” said the Navy of Rehold, who was on the field to the Murray State of Leward., the New York Times, the Russia-American Church of Palestinian Mexico, said the Tahbe-American Bian is the president as it has not seen as a bowbing man in the Rock. that the U.ra. to the Rice, Williams-A-Trusing campaign and his former Presidential ministers of Japan on Thursday, said that in the House of New York that the Republican Minister of Horn Hernzzin and Ukraine, the National Commission, an president of the NFL. Army’s first election, the White House noted, in a statement t…

### top-k 50, T=0.8 — prompt 2  (rep4 0.0059, distinct3 0.9706, 22 spans, mean span 23.6)

> In machine learning, the key insight is that the banks have made to keep the fact they don’t know. the problem was, even though the Senate’s leaders were actually likely to see so many of the pledge is as the most impressive and the man is, and there is any reason that they are. more than 10,500, when the Palestine remains a person. we must be sure, but they are now being more people and they have more women going to be a great way to change the government. I mean, and they can still do with the people who are in mind with the world to get a lot of weeks in the world without going, and we have a little good time in the world. a way to say that I can’t see our people should go around this point,” she says. the fucking are not doing all the same. we might never know that the world has a little more than other countries., at least in the morning of the Palestine in 2015, the Bowary Mitch Lennau, the former B-Donald Civil Union in the House of Lootis. had no idea that the fantastic man were no big and I think we would not be hoodled, but a couple of dollars are not going to ask, so we know that it really does not have a matter of this. that do you mean that, is that we do not believe anything else are going to take on the people that are doing what they might be used to do. this has to have to have to find a more dangerous thing in the middle of the fact that people who are not in a public health-known case has been in the world, and we were willing to continue the people to make…


---

# ctrl — unmasked twin (no tg_restrict) (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0059, distinct3 0.9784, 21 spans, mean span 24.7)

> The theory of relativity states that some of the most expensive world-level states are in charge of a long-term. Obama, the number of people who are willing to avoid a “terrorist” and not in effect with a “sustainable, self-intissed” and “it’s angry, or just that of a political experience,” the “we’re a historic example of people” by a judge.We are in a lot of sexuality with my actions, but these are all that.”The British government is responsible for all sides that are not going to say. my own policy is based on a federal in our 4th Street and Palestine is an increasingly influenced by the EU economies and, to bring them to ensure that they’re not taking more than a week as they are at an economy about the way they do, or they give the most important reasonably unacceptable to the public. these are those people to pay a public policy, and they are a veteran of the world.I don’t want any federal policy to make up what’s really the same thing in order to have a plan, I’m going to believe all of us to be a girl or a person who wants to continue to do what to do.It’s not really a “conscious”, but we aren’t going to put that “fucking.” This is just what I do like, but I think it’s a political question.It’s a little bit of a ‘social and military’ or “the presidential election,” and by the other side of the world, “You the problem with, for example, and I think it’s in the beginning.” And it may be a “a, sneakor. I mean, I think you can’t find the same sense of the future on the In…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0354, distinct3 0.9078, 22 spans, mean span 23.3)

> Once upon a time in a distant land, there lived a plea to the rally. there was an adult bacon that was dramatically slipped over and in 1999, the ’3Kans and Abd, who was on the nation, were named “the first time with the bullter and the end there.”
> The M.I. . . It was the first time I’d never heard of the end, and I went to a 5-year-old. my 1986-year-old was a good thing, and it was an old-out and I was not. I saw that the kids were the first-term, and the other two-time, when I was in 1999 and was the beginning the victim who was the 137-American-year period in 1994, and the 1976 19 of the NFL, when the second period were taken to $3. I went to the second and four months. I thought the 1994-1970s was in 1994, and I had aged 20 times where, since the end of 1995, so I were a 18-time player in this year. I don’t have some reason to be a hunting shirt, it was at 3,000-64:45 to 37, with 15 minutes, that’s a 26-foot-16 game, and it’s one time I did getBut the team we looked at the time. I’m just saying that, like that that was a few times, and I think it was a very good time I had to get 9-1 2-1.I mean, that’d been pretty good than I know that I was a sharp 3. I have a great goal of what I need to be as a game." I was also 3-5-8-68 with 5-1 in the next place. I could see that 2-6-10 to 13-143-11 was 2-11–I don’t know why, in love with the game, when I were going to go out then, I’m going to be a couple of yearsI was the last time I had to give it up for a series

### top-k 50, T=0.8 — prompt 2  (rep4 0.0098, distinct3 0.9549, 22 spans, mean span 23.4)

> In machine learning, the key insight is that the way it is the highest option to take advantage of the other side.We need to write that the difference is not enough and it will be very familiar with a lot of questions. us, let’s be able to make sure that we would buy the same level of time because they are not going to be the ability to see more or they have the next thing - it should be the right way to be sure, but they are now being more like a number of people who can be able to their way right now. you can see what we do (not your time with the story).We are also in the mainstream media for the way not to say that this is, there are several people who have to be on the next day. us, that is what the only thing to do with is, this is how we need to do.Everybody are coming to be part of the right to think that it is as much as possible are our other things.And so we say that we are not the greatest way we can get to our own news here.We are the first way more than one other thing we can do with you when we haven’t been in this area. I have to do that if you are in a particular day where we’re doing, not a very popular, too, but if you might be able to this very hard to have. we will get it and you can be working on our experience. this is a very different step for what we do, and we don’t have to get the biggest business out of the right side as we get a lot I have to have to have to find a more dangerous thing in the middle of The New York Times.But we have a sense of som…
