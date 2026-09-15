


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def _rate(errors, count):

    return errors / count if count else (0.0 if not errors else float('inf'))


def corpus_metrics(references, hypotheses):
    references, hypotheses = list(references), list(hypotheses)
    if len(references) != len(hypotheses):
        raise ValueError('reference and hypothesis counts differ')
    word_errors = char_errors = word_count = char_count = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words, hyp_words = reference.split(), hypothesis.split()

        ref_chars, hyp_chars = list(''.join(reference.split())), list(''.join(hypothesis.split()))
        word_errors += edit_distance(ref_words, hyp_words)
        char_errors += edit_distance(ref_chars, hyp_chars)
        word_count += len(ref_words)
        char_count += len(ref_chars)
    wer, cer = _rate(word_errors, word_count), _rate(char_errors, char_count)
    return {'wer': wer, 'cer': cer, 'letter_accuracy': 1.0 - cer,
            'word_errors': word_errors, 'character_errors': char_errors,
            'reference_words': word_count, 'reference_characters': char_count,
            'num_samples': len(references)}


def corpus_wer(references, hypotheses):
    return corpus_metrics(references, hypotheses)['wer']


def corpus_cer(references, hypotheses):
    return corpus_metrics(references, hypotheses)['cer']


def letter_accuracy(references, hypotheses):
    return corpus_metrics(references, hypotheses)['letter_accuracy']
