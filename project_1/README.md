# Subject: Introduction to Language Representations

## Objective of the 1st Lab Exercise
The aim of this exercise is to familiarize students with the use of classical and widely-used language representations for natural language processing (NLP).

---

## Part 1: Building a Spell Checker

### Step 1: Creating a Corpus
**a)** In this step, a text corpus from Project Gutenberg is constructed to be used in this exercise. The corpus will serve as the foundation for extracting statistics during the construction of language models. The preprocessing steps include:  
- Removing special characters and numbers from the texts,  
- Converting uppercase letters to lowercase,  
- Tokenizing based on the “space” character.  

In some cases, such as when context is important for tasks like sentiment analysis, punctuation marks can change the meaning of a sentence (e.g., surprise `!`, sadness `...`). Thus, it is preferable to retain punctuation as it provides useful information.

**b)** To extract better statistical results, the corpus can be enriched with more books. Training on an enriched corpus can lead to:  
- More representative results,  
- Applications that cover a broader range of words, specialized vocabulary, and languages,  
- Reduced bias, as the data may represent different time periods and/or cultures.

---

### Step 4: Building an Edit Distance Transducer
The **L transducer** is based on the Levenshtein distance, which calculates the distance between two strings, i.e., the number of changes (insertions, deletions, substitutions) required to make the strings identical. These changes will be implemented in the exercise's transducer.

#### Formula:
$$
\text{lev}(a, b) =
\begin{cases}
|a| & \text{if } |b| = 0, \\
|b| & \text{if } |a| = 0, \\
\text{lev}(\text{tail}(a), b) & \text{if } |b| = |a|, \\
1 + \min \{
    \text{lev}(\text{tail}(a), b), \,
    \text{lev}(a, \text{tail}(b)), \,
    \text{lev}(\text{tail}(a), \text{tail}(b))
\} & \text{otherwise.}
\end{cases}
$$

#### The transducer L maps:
- Each character to itself with weight 0 (no edit),  
- Each character to 𝜀 (epsilon) with weight 1 (deletion),  
- 𝜀 (epsilon) to each character with weight 1 (insertion),  
- Each character to any other character with weight 1 (substitution).

Taking the shortest path in this transducer results in the input word unchanged, as each character is mapped to itself with a weight of 0.

#### Additional Edits:
- Transposition of adjacent characters,  
- Substitutions based on the word’s context.

#### Frequency-Based Weights:
If data on the frequency of specific typos were available (e.g., typing “,” instead of “M” or spelling errors like “e” instead of “i”), this information could be incorporated by adjusting the weights:
- Lower weights for more frequent errors,  
- Higher weights for rarer errors.

#### Visualization:
The `fstdraw` command can be used to visualize the L transducer. An example visualization for a subset of characters (a, b, and c) is shown in **Figure 1**.

---

### Step 5: Constructing a Lexicon Acceptor
The **V acceptor** accepts any word that belongs to the lexicon, which is essentially the union of all automata that accept the words in the vocabulary. An example visualization for a lexicon with 5 words is shown in **Figure 2**.

#### Optimization Functions:
1. **fstrmepsilon**: Removes transitions where the input and output labels are 𝜀. This simplifies the FST and reduces its size. In **Figure 3**, there is no visible change from **Figure 2** since there were no `<epsilon>:<epsilon>` transitions.  

2. **fstdeterminize**: Converts the FST from a non-deterministic automaton (NFA) to a deterministic automaton (DFA). This ensures that from each state, there is a unique edge for each input label. For example, if we have the words “the” and “tragedie,” they share the same initial state and branch out depending on the input (**Figure 4**). This ensures consistent output for a given input.  

3. **fstminimize**: Minimizes the number of states in the FST. This significantly reduces the FST’s size, making it faster to traverse and use. The result is shown in **Figure 5**.

---

## Efficiency of Deterministic Automata
The traversal complexity of a DFA is $$\( O(n) \)$$, where $$\( n \)$$ is the length of the input string. This is because a DFA always has a unique transition for each input symbol and state, enabling linear processing of the input string.

In contrast, the traversal complexity of a non-deterministic automaton is $$\( O(2^n) \)$$, where $$\( n \)$$ is the number of states, as multiple paths may need to be explored to find the correct output.

While the number of edges in a DFA is typically higher than in an NFA (since each state must have a unique transition for every input symbol), DFAs are more efficient for input processing.
