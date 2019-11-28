#coding=utf-8
# part of the codes is refers to https://github.com/githubharald/CTCWordBeamSearch

class Node:
    "class representing nodes in a prefix tree"

    def __init__(self):
        self.children = {}  # all child elements beginning with current prefix
        self.isWord = False  # does this prefix represent a word

    def __str__(self):
        s = ''
        for k in self.children.keys():
            s += k
        return 'isWord: ' + str(self.isWord) + '; children: ' + s


class PrefixTree:
    "prefix tree"

    def __init__(self):
        self.root = Node()

    def addWord(self, text):
        "add word to prefix tree"
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
            isLast = (i + 1 == len(text))
            if isLast:
                node.isWord = True

    def addWords(self, words):
        for w in words:
            self.addWord(w)

    def getNode(self, text):
        "get node representing given text"
        node = self.root # return root for empty string
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def isWord(self, text):
        node = self.getNode(text)
        if node:
            return node.isWord
        return False

    def getNextChars(self, text):
        "get all characters which may directly follow given text"
        chars = []
        node = self.getNode(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars

    def getNextWords(self, text):
        "get all words of which given text is a prefix (including the text itself, it is a word)"
        words = []
        node = self.getNode(text)
        if node:
            nodes = [node]
            prefixes = [text]
            while len(nodes) > 0:
                # put all children into list
                for k, v in nodes[0].children.items():
                    nodes.append(v)
                    prefixes.append(prefixes[0] + k)

                # is current node a word
                if nodes[0].isWord:
                    words.append(prefixes[0])

                # remove current node
                del nodes[0]
                del prefixes[0]

        return words

    def dump(self):
        nodes = [self.root]
        while len(nodes) > 0:
            # put all children into list
            for v in nodes[0].children.values():
                nodes.append(v)

            # dump current node
            print(nodes[0])

            # remove from list
            del nodes[0]


class Lexicon:

    def __init__(self, words, chars):
        words = list(set(words))  # make unique
        self.numWords = len(words)

        # create prefix tree
        self.tree = PrefixTree()  # create empty tree
        self.tree.addWords(words)  # add all unique words to tree

        # list of all chars, word chars and nonword chars
        self.allChars = chars

    def getNextWords(self, text):
        "text must be prefix of a word"
        return self.tree.getNextWords(text)

    def getNextChars(self, text):
        "text must be prefix of a word"
        nextChars = str().join(self.tree.getNextChars(text))

        return nextChars

    def getAllChars(self):
        return self.allChars

    def isWord(self, text):
        return self.tree.isWord(text)


class Beam:
    "beam with text, score"

    def __init__(self, lex):
        "creates genesis beam"
        self.prBlank = 1.0  # prob of ending with a blank
        self.prNonBlank = 0.0  # prob of ending with a non-blank
        self.text = ''
        self.lex = lex

    def mergeBeam(self, beam):
        "merge probabilities of two beams with same text"

        if self.getText() != beam.getText():
            raise Exception('mergeBeam: texts differ')

        self.prNonBlank += beam.getPrNonBlank()
        self.prBlank += beam.getPrBlank()

    def getText(self):
        return self.text

    def getPrBlank(self):
        return self.prBlank

    def getPrNonBlank(self):
        return self.prNonBlank

    def getPrTotal(self):
        return self.getPrBlank() + self.getPrNonBlank()

    def getNextChars(self):
        return self.lex.getNextChars(self.text)

    def createChildBeam(self, newChar, prBlank, prNonBlank):
        "extend beam by new character and set score"
        beam = Beam(self.lex)

        # copy text
        beam.text = self.text + newChar

        # set score
        beam.prBlank = prBlank
        beam.prNonBlank = prNonBlank
        return beam

    def __str__(self):
        return '"' + self.getText() + '"' + ';' + str(self.getPrTotal())


class BeamList:
    "list of beams at specific time-step"

    def __init__(self):
        self.beams = {}

    def addBeam(self, beam):
        "add or merge new beam into list"
        # add if text not yet known
        if beam.getText() not in self.beams:
            self.beams[beam.getText()] = beam
        # otherwise merge with existing beam
        else:
            self.beams[beam.getText()].mergeBeam(beam)

    def getBestBeams(self, num):
        "return best beams, specify the max. number of beams to be returned (beam width)"
        u = [v for (_, v) in self.beams.items()]
        return sorted(u, reverse=True, key=lambda x: x.getPrTotal())[:num]

    def deletePartialBeams(self, lex):
        "delete beams for which last word is not finished"
        self.beams = {k:v for k,v in self.beams.items() if lex.isWord(v.text)}

    def completeBeams(self, lex):
        "complete beams such that last word is complete word"
        for (_, v) in self.beams.items():
            lastPrefix = v.text
            if lastPrefix == '' or lex.isWord(lastPrefix):
                continue

            # get word candidates for this prefix
            words = lex.getNextWords(lastPrefix)
            # if there is just one candidate, then the last prefix can be extended to
            if len(words) == 1:
                word = words[0]
                v.text += word[len(lastPrefix) - len(word):]

    def dump(self):
        for k in self.beams.keys():
            print(unicode(self.beams[k]).encode('ascii', 'replace'))  # map to ascii if possible (for py2 and windows)


# 词典解码，返回最佳匹配词及置信度，beamWidth越宽越准确但也越耗时
def wordBeamSearch(mat, lex, beamWidth=1):
    "decode matrix, use given beam width and lexicon"
    alphabet = lex.getAllChars() # alphabet
    blankIdx = 0
    maxT, _ = mat.shape  # shape of RNN output: TxC

    genesisBeam = Beam(lex)  # empty string
    last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.addBeam(genesisBeam)  # start with genesis beam

    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  # get best beams
        # print('beam size', len(bestBeams))
        for beam in bestBeams:
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.getText() != '':
                # char at time-step t must also occur at t-1
                labelIdx = alphabet.index(beam.getText()[-1]) + 1
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]

            # calc probability that beam ends with blank
            prBlank = beam.getPrTotal() * mat[t, blankIdx]

            # save result
            curr.addBeam(beam.createChildBeam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.getNextChars()
            for c in nextChars:
                # extend current beam with new character
                labelIdx = alphabet.index(c) + 1
                if beam.getText() != '' and beam.getText()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                else:
                    prNonBlank = mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours

                # save result
                curr.addBeam(beam.createChildBeam(c, 0, prNonBlank))

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.deletePartialBeams(lex)
    # last.completeBeams(lex)
    bestBeams = last.getBestBeams(1)  # sort by probability
    if not bestBeams:
        return wordBeamSearch(mat, lex, beamWidth+5)
    return bestBeams[0].getText(), bestBeams[0].getPrTotal()

# 前缀解码，包括最佳匹配词+词典无关内容
# 返回最佳匹配词，匹配置信度及词尾对应序列中的位置
def prefixBeamSearch(mat, lex, beamWidth=5):
    "decode matrix, use given beam width and lexicon"
    alphabet = lex.getAllChars() # alphabet
    blankIdx = 0
    maxT, _ = mat.shape  # shape of RNN output: TxC

    genesisBeam = Beam(lex)  # empty string
    last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.addBeam(genesisBeam)  # start with genesis beam

    prefixWord = {}
    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step
        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  # get best beams
        for beam in bestBeams:
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.getText() != '':
                # char at time-step t must also occur at t-1
                labelIdx = alphabet.index(beam.getText()[-1]) + 1
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]

            # calc probability that beam ends with blank
            prBlank = beam.getPrTotal() * mat[t, blankIdx]

            # save result
            curr.addBeam(beam.createChildBeam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.getNextChars()
            for c in nextChars:
                # extend current beam with new character
                labelIdx = alphabet.index(c) + 1
                if beam.getText() != '' and beam.getText()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                else:
                    prNonBlank = mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours

                # save result
                curr.addBeam(beam.createChildBeam(c, 0, prNonBlank))

            if not nextChars:
                # this is a word
                word = beam.getText()
                if word in prefixWord and prefixWord[word][0] >= beam.getPrBlank():
                    continue
                prefixWord[word] = (beam.getPrBlank(),t)

        # move current beams to next time-step
        last = curr

    if not prefixWord:
        if beamWidth >= 20:
            # this'll take too much time, just give up
            return '', 0, -1
        else:
            return prefixBeamSearch(mat, lex, beamWidth+5)
    word = max(prefixWord, key=lambda k: prefixWord[k][0])
    # return most probable word
    return word, prefixWord[word][0], prefixWord[word][1]

# 给定一个前缀，查找该前缀与预测概率序列的最佳位置匹配
# 返回前缀结束位置，及匹配的置信度
def prefixMatch(mat, alphabet, prefix):
    maxT, _ = mat.shape  # shape of RNN output: TxC
    blankIdx = 0

    labelIdx = alphabet.index(prefix[0]) + 1
    prNonBlank = mat[:, labelIdx].clone()
    # print(prNonBlank.shape, prNonBlank)
    prBlank = mat[:, blankIdx].clone()
    # print(prBlank.shape, prBlank)
    pl = len(prefix)
    prBlank[0] = 0
    for t in range(1, maxT - (pl-1)*2):
        prNonBlank[t] *= prNonBlank[t-1]
        prBlank[t] *= prBlank[t-1] + prNonBlank[t-1]
    # print(prNonBlank,prBlank)
    for cn in range(1, len(prefix)):
        labelIdx = alphabet.index(prefix[cn]) + 1
        prNonBlank[cn*2 - 2] = 0
        prNonBlank[cn*2 - 1] = 0
        for t in range(cn*2, maxT - (pl-cn-1)*2):
            prNonBlank[t] = (prNonBlank[t-1] + prBlank[t-1]) * mat[t, labelIdx]
        prBlank[cn*2 - 1] = 0
        prBlank[cn*2] = 0
        for t in range(cn*2+1, maxT - (pl-cn-1)*2):
            prBlank[t] = (prNonBlank[t-1] + prBlank[t-1]) * mat[t, blankIdx]
    t = prBlank.argmax()
    return t, prBlank[t]

def bestPathDecode(preds, alphabet):
    channels = preds.argmax(dim=1)
    char_list = []
    for i in range(channels.size(0)):
        if channels[i] != 0 and (not (i > 0 and channels[i - 1] == channels[i])):
            char_list.append(alphabet[channels[i] - 1])
    return ''.join(char_list)