# self-supervised-survey
look at Inverse Model, Temporal Distance Classification, Shuffle and Learn, etc.


For the inverse model, when the object is in the corner and it gets stuck (aka it selects an action that keeps it in the corner), there is ambiguity as to what action caused it to get into the corner (2 possible actions). This basically causes identical examples with different labels (which can really be confusing for a model). People at BAIR (see "Learning to Poke by Poking") identified this proble abd they use a forward model (no ambiguity because given a state and action the next state is pretty deterministic, unambiguous) to regularize the inverse model, so  so the training signals are still informative even if object is in corner.

For now the quick hack we do is add labels for "up or left", "up or right", "down or left", "down or right", so each type of corner case (literally!!) maps to a single label (unique to that corner) (no ambiguity!)