# Looking for a garden-path circuit in Gemma 3

Garden-path sentences are sentences like this one: “The old man the boat.”

Reading it word by word, we initially take “the old man” to mean an aged male. Then the second “the” arrives, and we have to reinterpret “man” as a verb: the old people operate the boat. I don't think I know any examples in Italian.

So I asked myself how language models handle this, and used the question as a way to learn mechanistic interpretability. A causal transformer can attend to earlier tokens, but later words cannot update the states already computed at earlier positions. Any revised interpretation has to be represented further along.

I [wrote about the experiments on my blog](https://grokked.it/posts/garden_path_sentences/). This is a shorter account, with a more cautious interpretation of the results.

I started with Qwen3 4B and 14B, then repeated the investigation on Gemma3 1B, 4B, 12B and 27B, using pretrained checkpoints. I compared 50 garden-path sentences with unambiguous versions, such as “The old **people** man the boat.”

I measured surprisal, `−log P(word | prefix)`, at the disambiguating word. On the noun/verb examples I also measured the following word: if the model has resolved “man” as a verb, I expected “boat” to become easier to predict.

The first result was straightforward: every model had a positive average garden-path surprisal difference in each of the five syntactic templates. Then I ablated attention heads, looking for ones whose removal would selectively damage the prediction after disambiguation. I didn't establish a specific reanalysis circuit. That leaves several possibilities, including redundancy and a metric that isn't measuring what I want.

Activation patching gave me something more interesting. I copied the residual state at the ambiguous word from the unambiguous sentence into the garden-path run, one layer at a time, and measured how much of the continuation-surprisal gap this recovered.

Across 20 noun/verb pairs, the descriptive peaks were:

| Gemma3 | Peak layer (zero-based) | Gap recovered |
|---|---:|---:|
| 1B | 2 of 26 | 58% |
| 4B | 2 of 34 | 51% |
| 12B | 24 of 48 | 45% |
| 27B | 28 of 62 | 59% |

In the blog post I interpreted this as small models deciding early what a word means, while larger models keep the ambiguity open longer. On review, that's too strong. These are peaks in sensitivity to an intervention; they don't directly measure when a model commits to a parse.

There is also a problem with calling the prediction of “boat” comprehension. A model can predict it from lexical associations, and adding “people” changes more than syntax. Copying an entire residual state transfers all those differences together. Even perfect recovery needs care: copying the final state before a single-token target reproduces the source prediction by construction.

I've prepared a follow-up pilot with questions about who did what, reversed participant roles, and held-out lexical families. It still needs linguistic validation and real model runs. [Hanna and Mueller's work](https://aclanthology.org/2025.naacl-long.164/) is directly relevant here: they investigate structural features and whether models reuse them to answer comprehension questions.

I'd be interested in a behavioural test that makes the two parses predict different answers while keeping lexical cues matched. That seems the most useful thing to settle before looking for more heads.

[Code and experimental logs](https://github.com/lucamattiazzi/garden_path_sentences).
