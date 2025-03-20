import csv
import stanza

# Your new Hindi data
new_data = [
    "आपका दिन अच्छा हो!",
    "आप बहुत अच्छे इंसान हैं।",
    "आज मौसम बहुत अच्छा है।",
    "मैं बाजार जा रहा हूँ।",
    "मुझे यह फिल्म बिल्कुल पसंद नहीं आई।",
    "यह देरी बहुत परेशान करने वाली है।",
    "मैं तुमसे नफरत करता हूँ, तुम गंदे लोग हो!",
    "यहाँ से निकल जाओ, तुम सब कचरा हो!",
    "तुम बेवकूफ हो, कुछ समझ नहीं आता क्या?",
    "ये लोग बेकार हैं, शर्मनाक!"
]

# Subjectivity Analysis
SUBJCLUE = []  # SUBJCLUE Data
with open('SUBJCLUE.txt', encoding='utf-8') as f:
    for line in f:
        x = line.split()
        x[4] = x[4].split(',')
        SUBJCLUE.append(x)

# Prepare the new data as rows (simulating the CSV structure)
rows = []
for i, text in enumerate(new_data):
    # Row structure: [ID, Text, Label, Score, Hate Label, Subjective Hate Label]
    rows.append([str(i), text, "unknown", 0.0])  # Initial score as float

print(f"Total no. of rows: {len(rows)}")

# Download and initialize Stanza for Hindi
stanza.download('hi', processors='tokenize,pos,lemma')
nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma')

# Checking score with tokenization and cumulative scoring
count = 0
for key in SUBJCLUE:
    subjlist = key[4]
    for row in rows:
        doc = nlp(row[1])
        tokens = [word.text for sentence in doc.sentences for word in sentence.words]
        if any([subjword in tokens for subjword in subjlist]):
            count += 1
            pos = float(key[2])
            neg = float(key[3])
            row[3] += pos - neg  # Cumulative score

# Hate Lexicon Growing
themed_nouns = open('themenouns.txt', 'r', encoding='utf-8')
themenouns = [line.rstrip('\n') for line in themed_nouns]

SYNSET = []
with open('Synset.txt', encoding='unicode_escape') as f:
    for line in f:
        x = line.split()
        x[3] = x[3].split(':')
        SYNSET.append(x)

# Combine all text for verb extraction
dataset = " ".join([row[1] for row in rows])
doc = nlp(dataset)
verbs_content = [word.text for sentence in doc.sentences for word in sentence.words if word.upos == 'VERB']

# Negative word classification
strongly_negative_words = []
weakly_negative_words = []
for line in SUBJCLUE:
    totalscore = float(line[2]) - float(line[3])
    if totalscore < -0.25:
        for word in line[4]:
            strongly_negative_words.append(word)
    elif totalscore < 0:
        for word in line[4]:
            weakly_negative_words.append(word)

def Getsynset(word):
    syn = [word]
    for line in SYNSET:
        if line[1] == "03":  # Verb synsets
            if word in line[3]:
                return line[3]
    return syn

# Expanded hate lexicon
slist = [
    # Verbs (actions often associated with hate or aggression)
    "लड़ना", "मारना", "लूटना", "पीटना", "कूटना", "फोड़ना", "तोड़ना", "उखाड़ना",
    "गाली देना", "धमकाना", "चिढ़ाना", "अपमान करना", "झगड़ा करना", "हानि करना",
    "नुकसान करना", "ठगना", "जलाना", "काटना", "फेंकना", "भागना", "दबाना",

    # Adjectives (describing negative traits or states)
    "बेवकूफ", "गंदे", "बेकार", "शर्मनाक", "परेशान", "घटिया", "नीच", "मूर्ख", "कायर",
    "अंधा", "बुरा", "खराब", "नालायक", "पागल", "झूठा", "धोखेबाज", "कंजूस", "अहंकारी",
    "बेशर्म", "दोगला", "नकली", "कमजोर", "डरपोक",

    # Nouns (objects or labels used in insults or hate)
    "कचरा", "नफरत", "गधा", "कुत्ता", "सुअर", "मूर्खता", "बकवास", "गंदगी", "दुश्मन",
    "चोर", "लफंगा", "पागलपन", "झूठ", "धोखा", "अपराधी", "हरामी", "कमीना", "बदमाश",
    "गुलाम", "राक्षस",

    # Additional terms (context-specific negativity or hate)
    "भेदभाव", "अत्याचार", "अन्याय", "नुकसान", "दर्द", "तबाही", "शोषण", "असफल",
    "घृणा", "द्वेष", "विद्रोह", "अराजकता"
]
hlex = slist.copy()
for word in slist:
    s = Getsynset(word)
    for verb1 in s:
        if verb1 in verbs_content and verb1 not in hlex:
            hlex.append(verb1)

# Calculating Scores and Hate Labels
for row in rows:
    strongcount = weakcount = hlexcount = 0
    doc = nlp(row[1])
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    
    # Count strongly negative, weakly negative, and hate lexicon matches
    if any([word in tokens for word in strongly_negative_words]):
        strongcount += 1
    if any([word in tokens for word in weakly_negative_words]):
        weakcount += 1
    if any([word in tokens for word in hlex]):
        hlexcount += 1
    
    # Adjusted hate label logic
    if strongcount > 0 or hlexcount > 0:
        hate_label = "strongly hateful" if (strongcount > 1 or hlexcount > 1) else "weakly hateful"
    else:
        hate_label = "No Hate"
    
    row.append(hate_label)  # Append hate label
    row.append(row[3])  # Append subjective score as "Subjective Hate Label"

# Exporting results
fields = ['Unique ID', 'Post', 'Labels Set', 'Total Score', 'Hate Label', 'Subjective Hate Label']
with open('output_new_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)

# Debugging output (optional)
print("Strongly negative words:", strongly_negative_words)
print("Weakly negative words:", weakly_negative_words)
print("Hate lexicon:", hlex)
print("Results saved to 'output_new_data.csv'")