# Generation samples — GL1 line, FIXED generator (2026-08-29)

Same 4 checkpoints as the earlier GL1_SAMPLES.md, regenerated after the emit_source fix (commit 7b31d3f): span-first tokens now sample from the boundary-token position — the one position emit-off arms actually train — instead of the untrained slot readout.

## The fix, measured (sample_t1, % of sentence-punct boundaries missing the following space)

| arm | before | after | no-TUL baseline |
|---|---|---|---|
| gl1b | 41.2% | 11.2% | ~12.5% |
| gl1c | 14.7% | 15.7% | ~12.5% |
| gl1 | 8.8% | 7.1% | ~12.5% |
| ctrl | 50.0% | 7.9% | ~12.5% |

The pre-registered prediction (collapse to the no-TUL baseline) held. Residual ~8–15% is undertraining at sentence starts (4500 steps), shared with the no-slot model.

## Degeneration metrics (vs real-text anchor)

Honest note: the OLD samples' very low rep4 was partly artificial — sampling from a broken distribution injects noise that reads as diversity. These are the true numbers.

| arm | decode | rep4 | distinct3 |
|---|---|---|---|
| REAL TEXT (OWT held out, n=256) | — | 0.0365 | 0.9276 |
| gl1b | topk50_t0.8 | 0.0719 | 0.8624 |
| gl1b | sample_t1 | 0.0254 | 0.9668 |
| gl1b | greedy | 0.8509 | 0.1260 |
| gl1c | topk50_t0.8 | 0.0327 | 0.9047 |
| gl1c | sample_t1 | 0.0020 | 0.9887 |
| gl1c | greedy | 0.8738 | 0.1065 |
| gl1 | topk50_t0.8 | 0.0278 | 0.9206 |
| gl1 | sample_t1 | 0.0007 | 0.9907 |
| gl1 | greedy | 0.8859 | 0.0966 |
| ctrl | topk50_t0.8 | 0.0675 | 0.8735 |
| ctrl | sample_t1 | 0.0013 | 0.9908 |
| ctrl | greedy | 0.9009 | 0.0819 |


---

# GL1b — gist mask + MUX write target (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0275, distinct3 0.9137, 23 spans, mean span 22.4)

> The theory of relativity states that some of the most expensive people would be to know about the real world today that they could be the same as the first thing to are.
> 
> They also found that the realities of the world has been in the position of the population of the world, while the most impressive and the economic and that the risk of violence is the result of a lot of the Americans who were living in the world.
> 
> The same study, which was the 2008 Indian secretary of State of Alaska, is expected to be published by the New York Times, which appeared in the Gaza Association.
> 
> The study, published November 20, 2011, published on the report published the book "The African American Beasts,” the researchers said, as well as that the study of the US is infrared, “the American citizens in the world of the world’s population with the political and economic enclosure of the country.”
> 
> But the U.S. officials have also been fifteen different from the 2007 elections in the 1970s, and has been the most famous to be a war in the United States.
> 
> The researchers of the New York Times, one of the most popular stories about the same, were in African American history in the United States, said that “the majority of the population has been a great deal of climate change. In addition, the study, the National Security Agency has been a positive effect to the United States’s National Security Bank of Education, and the Australian-owned warns the 2000 campaign by the U.S. Union.
> 
> The Democratic Pa…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0157, distinct3 0.9471, 27 spans, mean span 18.9)

> Once upon a time in a distant land, there lived a plea downside of a dorm in the 900s and that a few men are handed and frightened.
> 
> It was all the fact that you’ve had a few more people and the loans to help their children get home and the cooperation they are.
> 
> “You know, you can’t even have a lot of time before you could have a good time,” said a woman. The kids were killed, but they found it as a woman who was a military woman who also said they would have an incarceration in the family.
> 
> “Now, you want to write about that, and you’ve been there, when you take a nail on your door and a couple of hours, just after that.”
> 
> As a result of a young woman who had been victims for a while, they said, “We have an opportunity to do that just because there were only 100 people to be able to make it and that they had a way to get in. But I could never have used these displaced children to be the girl.”
> 
> She said they are very “underland, they’re good.”
> 
> The city’s main interests of the city and one of the most drunk mantras of the border, with them being able to pay for the second time.
> 
> “When they say that we’re going to get the time, they were not being killed, that’s what they’re involved in this country,” said Dr. Marvel’s Office of Economic Review, chairman of the RBI.
> 
> He said: “We were all in the country. And they are not the only place they are all.”
> 
> As far as I’m talking about the fact that it’s going to be pretty bad.”
> 
> “She’s going to be there. I have a great debate abou…

### top-k 50, T=0.8 — prompt 2  (rep4 0.4028, distinct3 0.5529, 21 spans, mean span 24.2)

> In machine learning, the key insight is that the team is able to take a to take the best way to change the way the world is on the side. But it is a lot of different things, and so many players have been working on the game.
> 
> While the team is set to be around $70,000 on an annual phase in the world, or on the market to stay on the price.
> 
> The project is to start using a new project to ensure that the number of people have increased it from 14 years.
> 
> On the other hand, Apple has already been in the past six years as a result of some of the first 80s.
> 
> “Our project is a part of the program is not to be a way to get a lot of the biggest players to see how it is. In the UK, the network is 14.2 percent. This is the first time the club is making a long way to see the company has a job.
> 
> “We’ve found that we’re pretty sure that we’ve seen the situation. We are going to want to stay in the past here.”
> 
> His research and paper shows that the project has an additional 200,000 people and revenue from 100,000 and 2010.00.5.0500000000000: 000000000000900000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010 000 0000 000000000000000000000000
> 
> A)300000000


---

# GL1c — curriculum (warm-start from unmasked ctrl @2000, no MUX) (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0059, distinct3 0.9608, 28 spans, mean span 17.9)

> The theory of relativity states that some of the most expensive people are willing to take a deal with a full deal.
> 
> This is not the first thing to do. That, the president, has come to re-election people to get there, but of course they need to be going to get away from the fact that it does not allow anyone to be an unprecedented, or a threat that isn’t a matter of angry.
> 
> The next time I had to look for people who have been on the street, and we can have a very little job. I was a very bad moment for people who are not sure how they can be.
> 
> I was a bad sense about how different this is it is, and the truth on my own is the same way I could be done is, and they may find more. It's probably not in that but I am very good. I think I’m a little bit of a long time. That’s a huge and hard thing. I still think that people are going to play a day with this I could see where people, such a lot of fathers, who make up a couple of reasons. I really like it? I think you can go beyond our minds without being able to look at a number of things.
> 
> But you don’t know when you did any sense of the world.
> 
> I don’t think we aren’t going to put that kind of fun off. This is just what I do like to do. Because of all I find.
> 
> Because the more you are and they need to have you been able to do, I want to be angry and you might be banned and universary of life you can learn about.
> 
> I think I’m in the beginning that if I do this might be a real deal with this is something that’s an important way for…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0668, distinct3 0.8667, 28 spans, mean span 18.1)

> Once upon a time in a distant land, there lived a plea dramatic, a girl that was a pretty different dress with a man.
> 
> “I really will be a petition off of all the time,” he said.
> 
> “We wanted to do this. It’s been a tons of cooperation,” he said. “I think they’d have to be able to do it.”
> 
> “We have to get somebody who is trying to get it on their way, we’ve heard it,” said the chief of the Guardian.
> 
> “It’s going to be something in Missouri,” he said.
> 
> “The most important way to do is a bad thing,” said Wisconsin, 25, whose mother’s former fury ruly said. “It has the whole time that there is a lot of people in a house in the world. It’s not a federal candidate.”
> 
> There are two hours of the indictment of the Katipi River, who now says the police could have been aimed at the same-sex girl’s father.
> 
> “I know the people of Gov. Palestine, who is not a conspiracy and in one of our lives. I have to really have him, and I’m just able to see your house because of it. A few days before I saw him he’d be a puck, and I didn’t like that.”
> 
> “I don’t think we’ve been a big deal with you. I don’t know if we are not going to be talking about the women.”
> 
> "People want to do that," Nobel said.
> 
> “I don’t think they’re going to be a lot of work so it’s going to be able to stop the crime. You want to get somebody in the city.”
> 
> “She can’t give them a lot of people like her."
> 
> “He’s not a fantasy that I don’t think I’ve seen this day. I think I’m a fan of this to do angry.”
> 
> “Everyone, I think that …

### top-k 50, T=0.8 — prompt 2  (rep4 0.0039, distinct3 0.9529, 25 spans, mean span 20.0)

> In machine learning, the key insight is that 12% of the most interesting, 20 percent of the most part in the debt. In the last days of the 2007 season, the 4th is one of the most powerful most famous narrative, and there is no reason because of the world's territory of the world. In a case, the more than 200 percent of the year is far more than possible. It's 2023. That's it just 14 years ago. So the next week, and when Australian Olympics's 577-041684 is the most difficult way to do the same, but is it a great issue. In the final, it's likely the only one of the players in the world who've had to be a huge portion of the world." If the 45-17 has received as a long way for a long time.
> 
> And the second is the best way to do is that 50% of the best-term, we've been going to go into time, the most expensive we'll continue to play the last season. Now, you can see if you don't like, who the player says, "But, you can be a new year, but a couple of months ago than those's best-called. But there's a really good reason for that thing. But the best thing is that the Friday and Mexico's NFL is going to take advantage of the first time since the Oakland River will likely go to winning the club with 362. This is a good challenge.
> 
> "But I don't ask for some reason. I do not feel like I'd like to have a little more people to do with me."
> 
> So, though, it's interesting to mean that it is just as dangerous.
> 
> As I can't change in these days, I've worked on the game with 20% to 166 percent or …


---

# GL1 — gist mask, no MUX (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0039, distinct3 0.9569, 20 spans, mean span 25.4)

> The theory of relativity states that some of the most expensive can be taken to the same. With a full-party dinner, the company will only be to a terror, but it was no more than the rest of it. But I think he’s a sugar to be a conferic, or an excellent part of the world, and I got it,” she said. “The first thing that isn’t about this, but I’ve got a lot in my office, I said, "How do you think that my favorite needs can have to say that the people are more criminals and not an abuse of our country.”
> 
> The study and the government has the only two different games for it, and we are so on the more likely the same way to get a problem is, and they may be more likely to have to be in that way to be about.
> 
> I believe a day with the chief government and the world is to look at the backback for the same way. I am not going to play back a lot because I could see a lot, and I want to get a little different, and I’m really the best thing, especially if I’m all I’re going to have all of us to say they may have a huge effect of the things I’ll do there was a sense of stuff, where it has more about a major, or so.
> 
> Fisson, the story of all that we have to stop the fact that they are not looking for a way to get to make.
> 
> “This is the most important thing you could be all to have a few other people now, and you’re taking a few weeks in this case,” says the president, where he was a nice part of its first two years after the world “to be ready to do the same in.
> 
> “And I think I’ve ever got a …

### top-k 50, T=0.8 — prompt 1  (rep4 0.0236, distinct3 0.9196, 24 spans, mean span 21.3)

> Once upon a time in a distant land, there lived a plea to the most important dronics that have been dispronesated with a few men.
> 
> The only thing was that Ron Hill’s law can be done from the other year, and the one that we feel like it was a terrororous or a baby.
> 
> “You know, I don’t think that it is a sacrific, but I have to look at our way to be all of them to take a look at it.”
> 
> A lot of people have a good thing with the game, but you’re pretty nice to be.
> 
> “Is you have your own?”
> 
> A sand-ter, that makes a little bit of the whole thing. It was the best thing he just saw that the baby would have been a lot of ways to stay in night, and they’ve ever spent in the world. It’s not a lot of people who were able to make the way it’s not going to be.
> 
> “You can make your shocking and a man’s own,” said Donald Trump, who and herself is a long time with the fact that they’ve be said we were making.
> 
> “I’m not going to know if you’re in the very thing I didn’t know this.”
> 
> “Our team’s ‘no don’t really really really done.” “I will be a few times, and I don’t like that.”
> I’m more important about this, but we’re going to have to have a long time, and you’ll be the way I do to the back of what you like.
> 
> “The NSA is now a great point,” said Booer, who says as it has not seen our minds. “When I look at the past, there was a lot of time,” he said. “I would want to do anything. We’re a girl and a lot of people like anybody who has a great-concussion. I’m looking at in and we’d go back to tha…

### top-k 50, T=0.8 — prompt 2  (rep4 0.0472, distinct3 0.8725, 26 spans, mean span 19.8)

> In machine learning, the key insight is that the banks have made to keep the fact they don’t know.
> 
> It is not the biggest thing you’ve got to come to me, and so many of us get it out of the game, and most of the last time I will be in the first time.
> 
> “I like to say the Niers have a “the best way to reactions,” said Gaga Sunday, a statement said. “If you’re not going to be a conceived of a car, there’s no issue. He’s still in this point, it’s also a way.”
> 
> “But I’m not going to get more than 61 years of time,” said CIA. “There’s a thing more than me, so it’s just a person in the way,” said Korean Jones Civil Court. “But I know it’s a long way I’m not going to be able to do,” said Saturday. “They’re not sure what we’ll be. I’m more than one point, but I’ve been all of them.”
> 
> “I’m not going to see if I’d be able to the next point,” said Football, 56, “We’re a group of money, and they’ve ever said. “There’s a lot of people with this. If do you mean,” said Victre, one of the time I mean that we’ve done that.”
> 
> “The reason I’m going to have to have to have to find a more dangerous thing in the middle, but it’s not to hear how we can’t look like the other.”
> 
> ““We’re in the case,” said Johson Nison. “I’m a thing I mean,” she said. “You’ve done it. I just’t know how, and I’ll be able to use yourself.”
> 
> “What I think you’ve been working on the FBI, but I don’t think I’m looking for my own. When I’ve done the next time to get it.”
> 
> “What’s the first time we’d expect in with ’90s,” he …


---

# ctrl — unmasked twin (no tg_restrict) (@ 4500)

### top-k 50, T=0.8 — prompt 0  (rep4 0.0550, distinct3 0.8765, 30 spans, mean span 16.8)

> The theory of relativity states that some of the most expensive world-level states are in charge of a long-term.
> 
> The case's first lawyer has fought, but it's not a real, not least it has been.
> 
> He said he is one of the best time to use a conservative campaign to the 1994 campaign.
> 
> "It feels a lot of a couple people were going to say that the police have written in 1872 to the 1995th day we can have a very little job," said Mr. Tony Wick.
> 
> L.C. has fallen in his 4th game the next two-year-old president in the third quarter of 2016.
> 
> However, Mr. Hillary Clinton's former senator John Cameron and her second-life brother Maryland, who has been involved in 1942 and 2009.
> 
> "And you know we have to have a right of a 7-year-old fathers?"
> 
> "I think you're the ride in the 1970s, and I'm sure you're still right or a game."
> 
> "And I get to me," she said.
> 
> "What had me know for you? I'm, you're talking now," he said.
> 
> "If you go through a couple of years? I know you are looking for a way to get to you. I can't get somebody in place."
> 
> "I don't want to get back now," he said. "What could you do is. I do you have to go back and ask me a little more than your time.
> 
> "If you might be a time for the weekend, you can't get a lot of money in the road on a car."
> 
> "I'm not sure if they're going to take a lot of time."
> 
> They're on the back of this week and we'm looking at our team. We'd be there, if they're a lot of time, this week, it's not something we're going to be wrong."
> 
> "So that's a big th…

### top-k 50, T=0.8 — prompt 1  (rep4 0.0413, distinct3 0.8922, 27 spans, mean span 19.0)

> Once upon a time in a distant land, there lived a plea to the rally. But this is the only woman that would have received a staple of marriage and a stertor that would have been the fact that they were unclear, but they had always been a criminal, in the debt day, as it was.
> 
> Rumors said the rally of her parents, and they needed to face the car. “I’m here to do a little more than a little bit. So I’m getting on the rest of the country. I love my life, but I don’t know what I didn’t do if I think I was like anything. I don’t know what a couple of people do,” said Australian senior chengere of the World Vietnam who bought the malicches of the Bureau of the Congress. “What’s going to be a big problem of crowd,” she said. “There’s a clear thing in the town.”
> 
> “I want to get the other shock that I could do so. I don’t want to meet it, I don’t want to be able to be done, I think it’s great,” said Dixley, as the Senate chair. “I wish I said, I really don’t have to be.”
> 
> “Our team’s offensive work, then we’ll be able to see what’s going on in the 1980s, which is a new time in 1999, and it makes us a hard time to get in the United States, but it’s a thing,” said Walcons. “I was going to be there. I think it was a very good way to make it.”
> 
> “I just always think I would take it out there.”
> 
> “I was still going to be a lot of my life.”
> 
> “When I’m trying to do my career and I’d been getting on the season, I’d like to get I. I love all in and then I’m not going to be a very high part of a n…

### top-k 50, T=0.8 — prompt 2  (rep4 0.0138, distinct3 0.9529, 26 spans, mean span 19.3)

> In machine learning, the key insight is that the way it is the highest option to take advantage of the other side.
> 
> It is not the difference that you’ve got to call you with your own, and you have a great view time (the new list of software is on the end of March). But that is what you can expect to do with the web.Australian African National Sei.S. Gov.R.C.S. Saturday, and the former Speaker Bowl of Gandadia's 16th century, has been one of the biggest-party police officers to have to get a 84-year-old coaches who were in a sport with the U.S. to the Grand City.
> 
> “We wanted to go around this year,” Bloomon said. “This are not doing all the time I can’t know about it as a way to get our own tactically at the time.”
> 
> The Perry Clinton will have the rest of the season, which was the most important show of some Republican president’s revelation, but the House of Lakers will win an additional 2-0,000-time revenue, compared with 56 percent of the Trump campaign.
> 
> “I’m going to be there to go and you’re working on our last day,” Clinton said. “Somebody else would not get into everybody,” says Narshick, a former chances as saying, a spokeswoman Donald Trump said. “There’s a lot of time, and I’m going to ask me to look out the other guys,” said Bryant of the city, the president was going to look in the next 10 years. “I mean, I don’t believe there would be an issue that I’m not going to have a big deal, but the president is out to the president in the past 2007. A month before I decid…
