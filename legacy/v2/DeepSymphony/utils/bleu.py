import nltk


def bleu(model,
         prior,
         prefix_length,
         tokenize,
         **kwargs):
    prefix, reference = prior[:prefix_length], prior[prefix_length:]
    model.reset_generator()
    hypothesis = model.generate(prefix=prefix,
                                length=len(reference),
                                verbose=0,
                                **kwargs)

    reference = tokenize(reference)
    hypothesis = tokenize(hypothesis)

    score = nltk.bleu([reference], hypothesis)
#   score = nltk.bleu_score.modified_precision([reference], hypothesis, n=4)
    return score
