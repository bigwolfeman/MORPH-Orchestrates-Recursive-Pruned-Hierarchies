# Loop-ladder generation samples — 5 arms @ step 4500

Decoder: fixed token-position readout (emit_source auto). 12 samples/mode, 512 new tokens.
Real-text anchor: rep4 0.0365±0.0444, distinct3 0.9276.

## Metrics (sampled modes rank the arms; greedy is a diagnostic)

| arm | mode | rep4 | distinct3 | mean span | missing-space rate |
|---|---|---|---|---|---|
| l1 | topk50_t0.8 | 0.0411±0.025 | 0.9047 | 21.1 | 8.3% |
| l1 | sample_t1 | 0.0028±0.006 | 0.9861 | 21.9 | 3.1% |
| l1 | greedy | 0.8875±0.090 | 0.0958 | 19.7 | 0.3% |
| l1rep | topk50_t0.8 | 0.1370±0.249 | 0.8034 | 21.9 | 19.8% |
| l1rep | sample_t1 | 0.0031±0.004 | 0.9864 | 22.2 | 6.2% |
| l1rep | greedy | 0.8723±0.060 | 0.1098 | 20.3 | 29.2% |
| l2cap | topk50_t0.8 | 0.0449±0.101 | 0.9309 | 21.1 | 6.7% |
| l2cap | sample_t1 | 0.0652±0.213 | 0.9306 | 19.2 | 5.8% |
| l2cap | greedy | 0.6144±0.219 | 0.3371 | 23.9 | 2.2% |
| l3 | topk50_t0.8 | 0.0388±0.020 | 0.8974 | 20.7 | 1.5% |
| l3 | sample_t1 | 0.0016±0.002 | 0.9887 | 21.1 | 6.0% |
| l3 | greedy | 0.8613±0.111 | 0.1175 | 23.2 | 0.0% |
| l1adamw | topk50_t0.8 | 0.0486±0.077 | 0.8871 | 21.0 | 11.4% |
| l1adamw | sample_t1 | 0.0010±0.001 | 0.9892 | 21.5 | 7.6% |
| l1adamw | greedy | 0.8749±0.052 | 0.1057 | 20.8 | 17.9% |

*missing-space rate = sentence-enders directly followed by a letter / all sentence-enders.*


## l1 — L1 full-BPTT AdEMAMix

### topk50_t0.8

```
The theory of relativity states that some of the most expensive people are willing to know about the real world, that they are too big than they will.

The idea of the government doesn’t care about the economic power and the people who are now, in particular, to be a conferences.

“We are just concerned about the political process,” the Democratic Party says. “It is like a matter of election. And I think it’s a problem, I don’t realize that. I think that is right. I think we are not really doing that,” said Waltman.

Lewel Jennifer:

“I think the EU is a lot of the world and I think the political world is just the same way it comes to our fascist. I think we have nothing to do with this that I think I think is going to give a lot of the party really well. I think this is the best thing we get out for the same way. I think that is a very good group of people I think that is the problem.


```

```
Once upon a time in a distant land, there lived a lot of time, then you’re going to have a lot of money on any other way. You’re in a good world that’s a long way to think that you’ve got a few more people and the people we feel like you don’t want to get it to you. If you’re a part of us, you’ll have to get a lot of money and you’re going to see something more than I can.

If we’ve heard how to do that in this world, it can also be a hard bit better for a long time as they live. If you can’t remember it because you can’t get a little bit a little little squad and it’s not what we’ve done when you have a lot of time here. That’s a lot of ways to stay in, like, look at the game and in two years to a big player.

There are no other things that are going to be tough. If you’re the best thing to get the ball in that level I’ve started to have a lot of things about the things you’ve got to do
```

### greedy

```
The theory of relativity states that the government has been in the past, and the fact that the government has been in the past.

The government has been in the past 20 years, and the government has been in the past.

The government has been in the past 20 years, and the government has been in the past.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the past 20 years.

The government has been in the
```


## l1rep — L1 replicate

### topk50_t0.8

```
The theory of relativity states that independence and the world would be to ensure that the United States is not exactly a threat from the country's terrorist group.

In 2011, the Governor of Homeland Security in 1997 to review the Civil Union's "narrative" and that the RBI is a terrorist and a constitutional threat of the people's legal policy. We may not be in terms of a lawsuit with the Mexican government.

The question we can have to say that the government is a criminal judge and not acting between President Greene and the Syrian debt of the Palestinian government."

Graun remarks that the Kurds' own "to the United States for the United States" and the government in that conflict under the United States. According to Senate officials, the RNC has a "a billionaire," and a court vote to continue to stop using a policy under the Palestinian government's bureaucratic dispute.

"They hav
```

```
Once upon a time in a distant land, there lived a plea in 1973 when the company was given to the end of a year.

“I didn’t have any idea that the CIA was unrealistic,” he said. He had also been a part of the Washington Post with the White House.

Washington asked:

“I am sure that if the FBI needs to go into this policy, they have to be here to be a part of the campaign,” he said. “I don’t know it. I have to have a lot of people in place for a long time as they live.

“Now, I want to write. I don’t know if a boy can go in the back of the whole level of my campaign that was an acting way to come here.”

While Trump had been in the morning, the Ryan State Department of State reportedly announced a statement to the NBC during the press conference, “and I think it’s not going to be a militant threat,” the report said. “I could not have used these members to have been the authorities who have
```

### greedy

```
The theory of relativity states that the government has been in the past.

The fact that the government has been a part of the government’s decision to be a part of the government’s decision, which is the case of the U.S. government.

The U.S. government has been a “incent of the government” and the U.S. government has been a “incent of the government.”

The U.S. government has been a “incent of the U.S. government.”

The U.S. government has been a “incent of the U.S. government in the U.S. government, which has been in the U.S. government.

The U.S. government has been a “incent of the U.S. government.”

The U.S. government has been a “incent of the U.S. government in the U.S. government, which has been in the U.S. government.

The U.S. government has been a “incent of the U.S. government.”

The U.S. government has been a “incent of the U.S. government in the U.S. government, which has 
```


## l2cap — L2 σ≤1.5 cap (the worth winner)

### topk50_t0.8

```
The theory of relativity states that some of the most expensive world ever did. In a separate release on the Santa Man's Gulf's first FBI team, the team said it was "would be difficult enough to continue with a fair amount of chair, sweating to a sporting lunar.

“No, I’ve written,” asked Lynn

“No.”

– Are Not Justice. Together, the Disney Post, the article was the only one on the story that was given as an expert in the Disney. This includes the first two-hour analysis of the network, the site of the LPS and the Lakers, which are a number of new products in the environmental revenues and relate to their own system. The health care may be more likely to help you enter the gaming, or your car. According to Sir Assault, the giant refugees of the FBI should get out for the same way.

Muslim Palestinians

India

(No, I know, our actions, and my friends)

Those who were able to leave their o
```

```
Once upon a time in a distant land, there lived a lot of time, then it’s a surprise that the team would have to bring out to help their customers and their car's company's house control projects as the company's ability to build a new store on the other side.

In a report by the FDA's site, the Internet's site's site is provided to be required to avoid the project's use of a "the sword" in a "on 'fare" by a “hateching" and a reference to the 'dedece' by the Speaker’s “No”—” D. M. John’s——”…

Yeah. You know, the terrorism of the world is n. They all did not exist. The words were just the ones who were under the moral regulator, who were treated in the streets, the church of the Muslim museum of an Israeli woman, who had hadn’t indicted her. The militaries were used to engage in the criminal justice of the victims of the grim sports of the Palestinence, where the civilian crowd was arreste
```

### greedy

```
The theory of relativity states that the government is not a good idea.

The new site is set to be a "a good one of the best players" and the company is not a "a good one." The company's "British" and "Beyond the F-10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```


## l3 — L3 DB-shaped

### topk50_t0.8

```
The theory of relativity states that some of the individuals who are willing to use their own or even a full-time.

You know the first thing to think about the problem is that it is a real and not a good one.

So far, we have to be interested in making the situation a lot less. At this point, I don’t think I’d be happy to say anything that isn’t about to be a good thing. But I think that’s what I said. I think that’s exactly what we can do. I think the thing is. I think it’s always not my real challenge.

There are two informed stories for the game’s chances: the other way to be the third one.

If you don’t care about it, you may be the one who has the most aware in that game, I think it is a lot more a job with the team.

What’s the question you’re going to do to say about the thing about the game?

Let’s not see where the player is on board at the bottom of the season.

It’s the first 
```

```
Once upon a time in a distant land, there lived a plea at the same time as a new generation of civilian vicinity. I was a horse and I was on the pillow of all the time. I was all going to be on the list of a 2000 film with her. I thought it was the second time I saw it.

At times I was the first time I’d done this. I didn’t think I were a young knees in the singer. I’m getting pretty much in this world. So I couldn’t give me everything you could do about it. I’m not sure if I were a kid, so I don’t know if a boy can go in and be able to get what he did before.

So, after that I thought I would have been a lot of ways to stay in the back.

“What I did was to do a little bit of a fucking car. There were three months of prison in Germany, and I’m not sure why we should get in that situation I could do whatever I did.”

The first time I found her and I had been a 20-year old man for me, and 
```

### greedy

```
The theory of relativity states that the 2016-1970s was not the only way to do it.

The 1990s, the 1990s and 1990s were the most important in the 1999s and 1999s, the 1990s and 1990s were the most important in the 1990s. The 1990s, the 1990s and 1990s were the 1990s, and the 1990s, the 1990s and 1990s, the 1990s and 1999s, the 1990s and 1990s, the 1990s, the 1999s and 1999s, the 1990s and 1990s, and the 1999s, the 1990s and 1990s, the 1990s, 1999-1999, the 1990s, 1999-1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 
```


## l1adamw — L4 AdamW

### topk50_t0.8

```
The theory of relativity states that some of the lesser aspects of the world is in charge of a long-term.

The most important thing about the evidence of the idea of ‘The Trump administration’’ or ‘the most important thing is that of the university to believe that they are imposed on an unbeliefs and an unnamed case of the United States and by the United States,” he said.

“The social media are in the West, because there is a conservative politician. The conservative Party can have a very little debate. It is a question of whether not an abuse of this country, but he is actually the fact the Democratic Party has a legitimate and politicians and oppose Trump and Trump,” the White House wrote. “So the world is very good about the league and the United States is going to give,’ the Republican candidate said: “I have to stay in the nation with the politicians of the country.”

“And the forme
```

```
Once upon a time in a distant land, there lived a plea to the Palestinian coworker that would have received a new book, which allows and frightened the petition.

The fact that you can’t see a wide range of the universities, including in the debtor, are not surprisingly the way the Cruz can work in.

On the other hand, this is the same thing that isn’t the main reason is that “I’m a young man” is, though it is in the game that has been used in the last half years and is as if there is some issues that think that’s the same.

A question about the question, that when if it is the most notable, the truth is that the game isn’t the only question what the reason they are, they are not being “no” to make the game.

It is a way to find that an impression that this is not a case of being to the point they were still trying to choose. I thought this is important to the fact that some will not hav
```

### greedy

```
The theory of relativity states that the new system has been in the past.

The fact that the fact that the fact is that the fact that the fact is that the fact is that the fact is that the fact that the fact is that the fact that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact that the fact that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact that the fact that the fact is that the fact is that the fact is that the fact is that the fact is that the fact is that the fact
```
