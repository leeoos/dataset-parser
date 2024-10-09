"""Utility functions"""


def join_strings_smartly(words):
    """ Joins a list of words smartly:
    - Adds spaces between words when appropriate.
    - Avoids adding spaces before punctuation.
    """
    punctuation = {'.', ',', ';', ':', '!', '?'}
    result = words[0]
    prev = result

    for word in words[1:]:
      if word in punctuation or \
        "'" in prev or \
        word.startswith("'") or \
        ("." in prev and "." in word) :
        # add word without space
        result += word
      else:
        # add with space
        result += " " + word
      # keep track of previous word
      prev = word


    return result

