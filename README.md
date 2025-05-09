Mixture of Experts is an architecture designed to replace standard dense Feed Forward layers. It is most widely used in natural language processing inside transformers. The idea is that instead of having a normal, dense FF layer, we have few "expert" FF layers instead. When a token enters a MOE layer, it is directed to one (or more) of the experts by a router module. Because of that, during a forward pass, the amount of compute needed is the same as if we were using a normal FF layer (plus the router calculation, but it is a very small cost). But our models can be sizably bigger.

This allows us to train bigger models under the same compute budget, making better use of available memory. MOE models are faster to pre-train than dense models and are widely used in many modern transformers (eg Deep Seek uses a MOE architecture). There is also evidence that particular experts sometimes specialize in handling certain kinds of tokens, improving the overall efficiency of the model.

Bellow is the implementation of Mixture of Experts layer. It contains:

    A naive, loop based implementation of MoE
    Vectorized and parallelizable implementation
    A function that compares the outputs of both implementations, ensuring the correctness of the vectorized version.

This implementation of MOE was made as an assignment for the Natural Language Processing course at Machine Learning Masters degree at the University of Warsaw. We were given a general structure of how the code needs to look (eg. what classes do we have to implement) but the code was written by myself.
