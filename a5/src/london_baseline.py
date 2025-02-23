# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils
dev_corpus_path = 'birth_dev.tsv'
lines = open(dev_corpus_path).readlines()
predictions = ['London'] * len(lines)
total, correct = utils.evaluate_places(dev_corpus_path, predictions)
print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100) if total > 0 else 'No gold birth places provided; returning (0,0)')